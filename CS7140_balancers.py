"""Debiasing binary predictions with linear programming.

Based on https://github.com/scotthlee/fairness

Binary implementation based on work by Hardt, Srebro, & Price (2016):
https://arxiv.org/pdf/1610.02413.pdf
"""

import pandas as pd
import numpy as np
import scipy as sp
import itertools
import seaborn as sns

from matplotlib import pyplot as plt
from itertools import combinations
from copy import deepcopy
from sklearn.metrics import roc_curve

import CS7140_tools


class BinaryBalancer:
    def __init__(self,
                 y,
                 y_,
                 a,
                 data=None,
                 summary=True,
                 threshold_objective='j'):
        """Initializes an instance of a PredictionBalancer.
        
        Parameters
        ----------
        y : array-like of shape (n_samples,) or str
            The true labels, either as a binary array or as a string \
            specifying a column in data. 
        
        y_ : array-like of shape (n_samples,) or str
            The predicted labels, either as an int (predictions) or float \
            (probabilities) array, or as a string specifying a column in data.
        
        a : array-like of shape (n_samples,) or str
            The protected attribute, either as an array, or as a string \
            specifying the column in data.
            
        data : pd.DataFrame instance, default None
            (Optional) DataFrame from which to pull y, y_, and a.
        
        summary : bool, default True
            Whether to print pre-adjustment false-positive and true-positive \
            rates for each group.
        
        threshold_objective : str, default 'j'
            Objective to use in evaluating thresholds when y_ contains \
            probabilities. Default is Youden's J index, or TPR - (1 - FPR) + 1.
        
        
        Attributes
        ----------
        actual_loss : float
            Loss on the current set of predictions in y_adj. 
        
        con : ndarray
            The coefficients of the constraint matrix for the linear program.
        
        goal : {'odds', 'opportunity'}, default 'odds'
            The fairness constraint to satisfy. Options are equalized odds \
            or equal opportunity. Set during .adjust().
        
        group_rates : dict
            The unadjusted tools.CLFRates object for each group.
        
        overall_rates : tools.CLFRates object
            The unadjusted CLFRates for the data overall.
        
        p : ndarray of shape (n_groups,)
            The proportions for each level of the protected attribute.
        
        pya : ndrray of shape (n_groups, 2)
            (P(y~=1 | y_=0), P(y~=1 | y_=1)) for each group after adjustment. \
            Set during .adjust().
        
        opt : SciPy.Optimize.OptimizeResult
            Optimizer solved by .adjust().
        
        rocs : list of sklearn.metrics.roc_curve results
            The roc curves for each group. Set only when y_ contains \
            probabilities.
        
        roc : tuple
            The theoretical optimum for (fpr, tpr) under equalized odds. Set \
            during .adjiust().
        
        theoretical_loss : float
            The theoretical optimum for loss given the constraints.
        
        y_adj : ndarray of shape (n_samples,)
            Predictions generated using the post-adjustment probabilities in \
            pya. Set on .adjust().
        """
        
        
        # Optional pull from a pd.DataFrame()
        if data is not None:
            y = data[y].values
            y_ = data[y_].values
            a = data[a].values
            
        # Setting the targets
        self.y = y
        self.y_ = y_
        self.a = a
        self.rocs = None
        self.roc = None
        self.con = None
        self.goal = None
        self.thr_obj = threshold_objective
        
        
        # Getting the group info
        self.groups = np.unique(a)
        group_ids = [np.where(a == g)[0] for g in self.groups]
        self.p = [len(cols) / len(y) for cols in group_ids]
        
        # Optionally thresholding probabilities to get class predictions
        if np.any([0 < x < 1 for x in y_]):
            print('Probabilities detected.\n')
            probs = deepcopy(y_)
            self.rocs = [roc_curve(y[ids], probs[ids]) for ids in group_ids]
            self.__roc_stats = [CS7140_tools.loss_from_roc(y[ids], 
                                                    probs[ids], 
                                                    self.rocs[i]) 
                              for i, ids in enumerate(group_ids)]
            if self.thr_obj == 'j':
                cut_ids = [np.argmax(rs['js']) for rs in self.__roc_stats]
                self.cuts = [self.rocs[i][2][id] for i, id in enumerate(cut_ids)]
                for g, cut in enumerate(self.cuts):
                    probs[group_ids[g]] = CS7140_tools.threshold(probs[group_ids[g]],
                                                          cut)
                self.y_ = probs.astype(np.uint8)
        
        # Calcuating the groupwise classification rates
        self.__gr_list = [CS7140_tools.CS_CLFRates(self.y[i], self.y_[i]) 
                         for i in group_ids]
        
        self.group_rates = dict(zip(self.groups, self.__gr_list))
        
        # And then the overall rates
        self.overall_rates = CS7140_tools.CS_CLFRates(self.y, self.y_)
        
        if summary:
            self.summary(adj=False)
        
    def adjust(self,
               goal='odds',
               round=4,
               return_optima=False,
               summary=True,
               binom=False):
        """Adjusts predictions to satisfy a fairness constraint.
        
        Parameters
        ----------
        goal : {'odds', 'opportunity'}, default 'odds'
            The constraint to be satisifed. Equalized odds and equal \
            opportunity are currently supported.
        
        round : int, default 4
            Decimal places for rounding results.
        
        return_optima: bool, default True
            Whether to reutn optimal loss and ROC coordinates.
        
        summary : bool, default True
            Whether to print post-adjustment false-positive and true-positive \
            rates for each group.
        
        binom : bool, default False
            Whether to generate adjusted predictions by sampling from a \
            binomial distribution.
        
        Returns
        -------
        (optional) optima : dict
            The optimal loss and ROC coordinates after adjustment.    
        """
        
        self.goal = goal
        
        # Getting the coefficients for the objective
        dr = [(g.nr * self.p[i], g.pr * self.p[i]) 
              for i, g in enumerate(self.__gr_list)]
        
        # Getting the overall error rates and group proportions
        s = self.overall_rates.acc
        e = 1 - s
        
        # Setting up the coefficients for the objective function
        obj_coefs = np.array([[(s - e) * r[0], 
                               (e - s) * r[1]]
                             for r in dr]).flatten()
        obj_bounds = [(0, 1)]
        
        # Generating the pairs for comparison
        n_groups = len(self.groups)
        group_combos = list(combinations(self.groups, 2))
        id_combos = list(combinations(range(n_groups), 2))
        
        # Pair drop to keep things full-rank with 3 or more groups
        if n_groups > 2:
            n_comp = n_groups - 1
            group_combos = group_combos[:n_comp]
            id_combos = id_combos[:n_comp]
        
        col_combos = np.array(id_combos) * 2
        n_pairs = len(group_combos)
        
        # Making empty matrices to hold the pairwise constraint coefficients
        tprs = np.zeros(shape=(n_pairs, 2 * n_groups))
        fprs = np.zeros(shape=(n_pairs, 2 * n_groups))
        
        # Filling in the constraint matrices
        for i, cols in enumerate(col_combos):
            # Fetching the group-specific rates
            gc = group_combos[i]
            g0 = self.group_rates[gc[0]]
            g1 = self.group_rates[gc[1]]
            
            # Filling in the group-specific coefficients
            tprs[i, cols[0]] = g0.fnr
            tprs[i, cols[0] + 1] = g0.tpr
            tprs[i, cols[1]] = -g1.fnr
            tprs[i, cols[1] + 1] = -g1.tpr
            
            fprs[i, cols[0]] = g0.tnr
            fprs[i, cols[0] + 1] = g0.fpr
            fprs[i, cols[1]] = -g1.tnr
            fprs[i, cols[1] + 1] = -g1.fpr
        
        # Choosing whether to go with equalized odds or opportunity
        if 'odds' in goal:
            self.con = np.vstack((tprs, fprs))
        elif 'opportunity' in goal:
            self.con = tprs
        elif 'parity' in goal:
            pass
        
        con_b = np.zeros(self.con.shape[0])
        
        # Running the optimization
        self.opt = sp.optimize.linprog(c=obj_coefs,
                                       bounds=obj_bounds,
                                       A_eq=self.con,
                                       b_eq=con_b,
                                       method='highs')
        self.pya = self.opt.x.reshape(len(self.groups), 2)
        
        # Setting the adjusted predictions
        self.y_adj = CS7140_tools.pred_from_pya(y_=self.y_, 
                                         a=self.a,
                                         pya=self.pya, 
                                         binom=binom)
        # Getting theoretical (no rounding) and actual (with rounding) loss
        self.actual_loss = 1 - CS7140_tools.CS_CLFRates(self.y, self.y_adj).acc
        cmin = self.opt.fun
        tl = cmin + (e*self.overall_rates.nr) + (s*self.overall_rates.pr)
        self.theoretical_loss = tl
        
        # Calculating the theoretical balance point in ROC space
        p0, p1 = self.pya[0][0], self.pya[0][1]
        group = self.group_rates[self.groups[0]]
        fpr = (group.tnr * p0) + (group.fpr * p1)
        tpr = (group.fnr * p0) + (group.tpr * p1)
        self.roc = (np.round(fpr, round), np.round(tpr, round))
        
        if summary:
            self.summary(org=False)
        
        if return_optima:                
            return {'loss': self.theoretical_loss, 'roc': self.roc}
    
    def predict(self, y_, a, binom=False):
        """Generates bias-adjusted predictions on new data.
        
        Parameters
        ----------
        y_ : ndarry of shape (n_samples,)
            A binary- or real-valued array of unadjusted predictions.
        
        a : ndarray of shape (n_samples,)
            The protected attributes for the samples in y_.
        
        binom : bool, default False
            Whether to generate adjusted predictions by sampling from a \
            binomial distribution.
        
        Returns
        -------
        y~ : ndarray of shape (n_samples,)
            The adjusted binary predictions.
        """
        # Optional thresholding for continuous predictors
        if np.any([0 < x < 1 for x in y_]):
            group_ids = [np.where(a == g)[0] for g in self.groups]
            y_ = deepcopy(y_)
            for g, cut in enumerate(self.cuts):
                y_[group_ids[g]] = CS7140_tools.threshold(y_[group_ids[g]], cut)
        
        # Returning the adjusted predictions
        adj = CS7140_tools.pred_from_pya(y_, a, self.pya, binom)
        return adj
    
    def plot(self, 
             s1=50,
             s2=50,
             preds=False,
             optimum=True,
             roc_curves=True,
             lp_lines='all', 
             shade_hull=True,
             chance_line=True,
             palette='colorblind',
             style='white',
             xlim=(0, 1),
             ylim=(0, 1),
             alpha=0.5):
        """Generates a variety of plots for the PredictionBalancer.
        
        Parameters
        ----------
        s1, s2 : int, default 50
            The size parameters for the unadjusted (1) and adjusted (2) ROC \
            coordinates.
        
        preds : bool, default False
            Whether to observed ROC values for the adjusted predictions (as \
            opposed to the theoretical optima).
        
        optimum : bool, default True
            Whether to plot the theoretical optima for the predictions.
        
        roc_curves : bool, default True
            Whether to plot ROC curves for the unadjusted scores, when avail.
        
        lp_lines : {'upper', 'all'}, default 'all'
            Whether to plot the convex hulls solved by the linear program.
        
        shade_hull : bool, default True
            Whether to fill the convex hulls when the LP lines are shown.
        
        chance_line : bool, default True
            Whether to plot the line ((0, 0), (1, 1))
        
        palette : str, default 'colorblind'
            Color palette to pass to Seaborn.
        
        style : str, default 'dark'
            Style argument passed to sns.set_style()
        
        alpha : float, default 0.5
            Alpha parameter for scatterplots.
        
        Returns
        -------
        A plot showing shapes were specified by the arguments.
        """
        # Setting basic plot parameters
        plt.xlim(xlim)
        plt.ylim(ylim)
        sns.set_theme()
        sns.set_style(style)
        cmap = sns.color_palette(palette, as_cmap=True)
        
        # Plotting the unadjusted ROC coordinates
        orig_coords = CS7140_tools.group_roc_coords(self.y, 
                                             self.y_, 
                                             self.a)
        sns.scatterplot(x=orig_coords.fpr,
                        y=orig_coords.tpr,
                        hue=self.groups,
                        s=s1,
                        palette='colorblind')
        plt.legend(loc='lower right')
        
        # Plotting the adjusted coordinates
        if preds:
            adj_coords = CS7140_tools.group_roc_coords(self.y, 
                                                self.y_adj, 
                                                self.a)
            sns.scatterplot(x=adj_coords.fpr, 
                            y=adj_coords.tpr,
                            hue=self.groups,
                            palette='colorblind',
                            marker='x',
                            legend=False,
                            s=s2,
                            alpha=1)
        
        # Optionally adding the ROC curves
        if self.rocs is not None and roc_curves:
            [plt.plot(r[0], r[1]) for r in self.rocs]
        
        # Optionally adding the chance line
        if chance_line:
            plt.plot((0, 1), (0, 1),
                     color='lightgray')
        
        # Adding lines to show the LP geometry
        if lp_lines:
            # Getting the groupwise coordinates
            group_rates = self.group_rates.values()
            group_var = np.array([[g]*3 for g in self.groups]).flatten()

            # Getting coordinates for the upper portions of the hulls
            upper_x = np.array([[0, g.fpr + 0.001, 1] for g in group_rates]).flatten() #g.fpr + 0.001
            upper_y = np.array([[0, g.tpr, 1] for g in group_rates]).flatten()
            upper_df = pd.DataFrame((upper_x, upper_y, group_var)).T
            upper_df.columns = ['x', 'y', 'group']
            upper_df = upper_df.astype({'x': 'float',
                                        'y': 'float',
                                        'group': self.groups.dtype})
            # Plotting the line
            sns.lineplot(x='x', 
                         y='y', 
                         hue='group', 
                         data=upper_df,
                         alpha=0.75, 
                         legend=False)
            
            # Optionally adding lower lines to complete the hulls
            if lp_lines == 'all':
                lower_x = np.array([[0, 1 - g.fpr - 0.001, 1] #just a little hack for when fpr = 0
                                    for g in group_rates]).flatten()
                lower_y = np.array([[0, 1 - g.tpr, 1] 
                                    for g in group_rates]).flatten()
                lower_df = pd.DataFrame((lower_x, lower_y, group_var)).T
                lower_df.columns = ['x', 'y', 'group']
                lower_df = lower_df.astype({'x': 'float',
                                            'y': 'float',
                                            'group': self.groups.dtype})
                # Plotting the line
                sns.lineplot(x='x', 
                             y='y', 
                             hue='group', 
                             data=lower_df,
                             alpha=0.75, 
                             legend=False)

            # Shading the area under the lines
            if shade_hull:
                for i, group in enumerate(self.groups):
                    uc = upper_df[upper_df.group == group]
                    u_null = np.array([0, uc.x.values[1], 1])
                    if lp_lines == 'upper':
                        plt.fill_between(x=uc.x,
                                         y1=uc.y,
                                         y2=u_null,
                                         color=cmap[i],
                                         alpha=0.2) 
                    if lp_lines == 'all':
                        lc = lower_df[lower_df.group == group]
                        l_null = np.array([0, lc.x.values[1], 1])
                        plt.fill_between(x=uc.x,
                                         y1=uc.y,
                                         y2=u_null,
                                         color=cmap[i],
                                         alpha=0.2) 
                        plt.fill_between(x=lc.x,
                                         y1=l_null,
                                         y2=lc.y,
                                         color=cmap[i],
                                         alpha=0.2)        
        
        # Optionally adding the post-adjustment optimum
        if optimum:
            if self.roc is None:
                print('.adjust() must be called before optimum can be shown.')
                pass
            
            elif 'odds' in self.goal:
                plt.scatter(self.roc[0],
                                self.roc[1],
                                marker='x',
                                color='black')
                print('opt coords ', self.roc[0],' ',self.roc[1])
            
            elif 'opportunity' in self.goal:
                plt.hlines(self.roc[1],
                           xmin=0,
                           xmax=1,
                           color='black',
                           linestyles='--',
                           linewidths=0.5)
            
            elif 'parity' in self.goal:
                pass
        
        plt.show()
        
    def summary(self, org=False, adj=False):
        """Prints a summary with FPRs and TPRs for each group.
        
        Parameters:
            org : bool, default True
                Whether to print results for the original predictions.
            
            adj : bool, default True
                Whether to print results for the adjusted predictions.
        """
        
        
        #DATA COLLECTION
        pr = [] #Positive rate for Y
        pr_org = [] #Positive rate for Y_
        pr_adj = [] #Positive rate for Y*
        
        org_loss = [] #Loss on (Y,Y_)
        adj_loss = [] #Loss on (Y,Y*)
        
        for j in range(len(self.groups)):
            yA = np.array([val for i, val in enumerate(self.y) if self.a[i] == self.groups[j]])
            yA_1 = np.array([val for i, val in enumerate(yA) if yA[i] == 1])
            pr.append(len(yA_1)/len(yA))
            
            yA_org = np.array([val for i, val in enumerate(self.y_) if self.a[i] == self.groups[j]])
            pr_org.append(CS7140_tools.CS_CLFRates(yA, yA_org).pr)
            org_loss.append(1 - CS7140_tools.CS_CLFRates(yA, yA_org).acc)
            
            yA_adj = np.array([val for i, val in enumerate(self.y_adj) if self.a[i] == self.groups[j]])
            pr_adj.append(CS7140_tools.CS_CLFRates(yA, yA_adj).pr)
            adj_loss.append(1 - CS7140_tools.CS_CLFRates(yA, yA_adj).acc)
            
            
        
        #PRE AND POST ADJUSTMENT GROUP RATES
        if org:
            org_coords = CS7140_tools.group_roc_coords(self.y, self.y_, self.a)
            print('\nPre-adjustment group rates are \n')
            print(org_coords.to_string(index=False))
            
        
        if adj:
            adj_coords = CS7140_tools.group_roc_coords(self.y, self.y_adj, self.a)
            print('\nPost-adjustment group rates are \n')
            print(adj_coords.to_string(index=False))
        

        
        #CALIBRATION
        calibration_indx = ['Fraction where Y_ = 1', 'Fraction where Y* = 1','Fraction where Y = 1']
        calibration = pd.DataFrame([np.array(pr_org), np.array(pr_adj),np.array(pr)],index=calibration_indx, columns = self.groups)
        print('\nCalibration: \n')
        print(calibration.head())
        print('\n')
        
        
        
        #LOSS
        org_loss.append(1 - self.overall_rates.acc)
        adj_loss.append(1 - CS7140_tools.CS_CLFRates(self.y, self.y_adj).acc)
        loss_indx = ['Y_','Y*']
        
        loss_cols = np.concatenate([self.groups, np.array(['overall'])])
        loss = pd.DataFrame([org_loss, adj_loss],index=loss_indx, columns=loss_cols)
        print('\nLoss: \n')
        print(loss.head())
        print('\n')
        
        
        loss_data = {'predictor':['Y_','Y_','Y_','Y*','Y*','Y*'],
                'group':[self.groups[0],self.groups[1],'overall',self.groups[0],self.groups[1],'overall'],
                'loss':np.concatenate([org_loss,adj_loss])}
        df = pd.DataFrame(loss_data)

        sns.barplot(data=df, x="group", y="loss",hue='predictor')
        plt.title('Comparing Loss')
        plt.xlabel('Group')
        plt.ylabel('Loss')
        plt.legend(loc='lower right')
        plt.show()
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            