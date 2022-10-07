import Rbeast as rb

class beastmaster():
    def __init__(self,
                 startTime = 1871.0, 
                 deltaTime = 1.0,  
                 isRegularOrdered = True, 
                 time = None, 
                 whichDimIsTime = 1, 
                 freq = None,
                 missingValue = float('nan'), 
                 season = 'none',  
                 maxMissingRate = 0.7500,
                 detrend        = False,
                 deseasonalize  = False,
                 sorder_minmax = [0,5], 
                 scp_minmax = [0,10], 
                 sseg_minlength = None,
                 torder_minmax = [0,1],
                 tcp_minmax  = [0,10], 
                 tseg_minlength  = None, 
                 precValue = 1.5, 
                 precPriorType = 'uniform', 
                 seed = 9543434, 
                 samples = 3000,
                 thinningFactor = 3, 
                 burnin = 150, 
                 chainNumber  = 3, 
                 maxMoveStepSize  = 4, 
                 trendResamplingOrderProb = 0.100, 
                 seasonResamplingOrderProb = 0.1700, 
                 credIntervalAlphaLevel = 0.950, 
                 dumpInputData = False,
                 whichOutputDimIsTime = 1, 
                 computeCredible = False, 
                 fastCIComputation = True, 
                 computeSeasonOrder = False, 
                 computeTrendOrder = False, 
                 computeSeasonChngpt = True, 
                 computeTrendChngpt = True, 
                 computeSeasonAmp = False, 
                 computeTrendSlope = False,
                 tallyPosNegSeasonJump= False, 
                 tallyPosNegTrendJump = False, 
                 tallyIncDecTrendJump = False, 
                 hasOutlier = False,
                 printProgressBar = True, 
                 printOptions = True, 
                 consoleWidth = 0, 
                 numThreadsPerCPU = 1, 
                 numParThreads = 2
                ):
    
        """
        ###############################################################################################################
        # Desription
        ###############################################################################################################
        BEAST (Bayesian Estimator of Abrupt change, Seasonality, and Trend) is a fast, generic Bayesian model averaging 
        algorithm to decompose time series or 1D sequential data into individual components, such as abrupt changes, trends, 
        and periodic/seasonal variations, as described in Zhao et al. (2019). BEAST is useful for changepoint detection 
        (e.g., breakpoints, structural breaks, regime shifts, or anomalies), trend analysis, time series decomposition 
        (e.g., trend vs seasonality), time series segmentation, and interrupted time series analysis.

        Zhao, K., Wulder, M. A., Hu, T., Bright, R., Wu, Q., Qin, H., Li, Y., Toman, E., Mallick B., Zhang, X., & Brown, M. (2019).
        Detecting change-point, trend, and seasonality in satellite time series data to track abrupt changes and nonlinear dynamics: 
        A Bayesian ensemble algorithm. Remote Sensing of Environment, 232, 111181. (the BEAST paper)

        ###############################################################################################################
        # Parameters
        ###############################################################################################################      
        # Metadata Parameters
        --------------------------------------------
        
        If present, metadata parameters can be either an INTEGER specifying the known period of the cyclic/seasonal component
        or a LIST specifying various parameters to describe the 1st argument Y. If missing, default values will be used, but 
        metadata paramaeters must be explicitly specified if the input Y is a 2D matrix or 3D array. metadata is not part of 
        BEAST's Bayesian formulation but just some additional info to interpret Y. Below are possible fields in metadata; 
        not all of them are always needed, depending on the types of inputs (e.g., 1D, 2D or 3D; regular or irregular).
        
        startTime: float
            number (default to 1.0) or Date; the time of the 1st datapoint of y. It can be specified as a scalar (e.g., 2021.0644)
            , a vector of three values in the order of Year, Month, and Day (e.g., [2021,1,24] )
        deltaTime: float
            specifies the time interval between consecutive data points. It is optional for regular data 
            (default to 1.0 if not supplied), but has to be specified for irregular data because deltaTime is needed 
            to aggregate/resample the irregular time series into regular ones.
        isRegularOrdered: bool
            If TRUE, the dataset is assumed to be regular and if FALSE,dataset is irregular or regular but unordered 
            in time.           
        time: array
            array specifying the timesteps for irregular time series datasets. Needed ONLY if isRegularOrdered = FALSE 
            (i.e. irregular inputs). Ignored if isRegularOrdered = TRUE (i.e., regular data for which startTime and deltaTime 
            are used instead).
             - a vector of numerical values to indicate times. The unit of the times is irrelevant to BEAST as long as it remains 
               consistent as the unit used for specifying other variables such as startTime and deltaTime.
               The length of the time must be the same as the datasets time dimension.
        whichDimIsTime
               specifies the which dimension of a multideimensional dataset is time. For example, whichDimIsTime = 1 for a 
               190x35 2D input indicates 35 time series of length 190 each; whichDimIsTime = 2 for a 100x200x300 3D input 
               indicates 30000=100*300 time series of length 200 each.
        freq: integer. 
            Needed only for data with a periodic/cyclic component (i.e., season='harmonic' or 'dummy' ) and ignored for 
            trend-only data (i.e., season='none'). The "freq" parameter must be an INTEGER specifying 
            the number of samples/values/points per cycle (e.g, a monthly time series with an annual period has a frequency of 12. 
            If freq is absent, BEAST first attempts to guess its value via auto-correlation before fitting the model.    
        missingValue: numeric 
            a customized value to indicate bad/missing values in the time series, in additon to those NA or NaN values.
        season: str
            'none':     dataset is trend-only; no periodic components are present in the time series. The args for the seasonal component
                        (i.e.,sorder.minmax, scp.minmax and sseg.max) will be irrelevant and ignored.
            'harmonic': dataset has a periodic/seasonal component. The term season is a misnomer, being used here to broadly refer to any 
                        periodic variations present in y. The periodicity is NOT a model parameter estimated by BEAST but a known constant given
                        by the user through freq. By default, the periodic component is modeled as a harmonic curve a combination of sins and cosines.
            'dummy':    the same as 'harmonic' except that the periodic/seasonal component is modeled as a non-parametric curve. The harmonic order 
                        arg sorder.minmax is irrelevant and is ignored.
        deseasonalize: bool
            if true, the input ts is first de-seasonalize by removing a global seasonal component, prior to applying BEAST
        detrend: bool
            if true, the input ts is first de-trended by removing a global trend component, prior to applying BEAST

        # Hyperprior Paramaters
        --------------------------------------------

        Hyperprior parameters in the Bayesian formulation of the BEAST model. Because they are part of the model, the fitting 
        result may be sensitive to the choices of these hyperparameters. 
        Below are possible parameters:     
        
        sorder_minmax: list
                The min and max harmonic orders considered to fit the seasonal component. s
                seasonMinMaxOrder is only used if the time series has a seasonal component (i.e., season='harmonic') and ignored for 
                trend-only data or when season='dummy'. If min(seasonMinMaxOrder) == max(seasonMinMaxOrder) , BEAST assumes a constant 
                harmonic order used and won't infer the posterior probability of harmonic orders.
        scp_minmax: list
            the min and max number of seasonal changepints allowed in segmenting and fitting the seasonal component. 
            seasonMinMaxKnotNum is only used if the time series has a seasonal component (i.e., season='harmonic' or season='dummy')
            and ignored for trend-only data. If "minimum order" == "maximum order", BEAST assumes a constant number of
            changepoints and won'tinfer the posterior probability of the number of changepoints, but it will still estimate
            the occurance probability of the changepoints over time (i.e., the most likely times at which these changepoints occur). If 
            min(seasonMinMaxOrder)=max(seasonMaxOrder)=0, no changepoints are allowed in the seasonal component; then a global harmonic model 
            is used to fit the seasonal component.
        sseg_minlength: int
            he min seperation time between two neighboring season changepoints. That is, when fitting a piecewise harmonic seasonal 
            model, no two changepoints are allowed to occur within a time window of seasonMinSepDist. seasonMinSepDist must be an 
            unitless integer–the number of time intervals/data points so that the time window in the original unit is seasonMinSepDist*deltaTime.
        torder_minmax: list
            he min and max orders of the polynomials considered to fit the trend component. The zero-th order corresponds to a constant 
            term/ a flat line and the 1st order is a line. If min(trendMinMaxOrder) = max(trendMinMaxOrder), BEAST assumes a constant 
            polynomial order used and won't infer the posterior probability of polynomial orders.
        tcp_minmax: list
            the min and max number of trend changepoints allowed in segmenting and fitting the trend component. 
            If min(trendMinMaxOrder) = max(trendMinMaxOrder), BEAST assumes a constant number of changepoints in the fitted trend and won't infer
            the posterior probability of the number of trend changepoints, but it will still estimate the occurrence probability of the 
            changepoints over time (i.e., the most likely times at which these changepoints occur). If trendMinOrder=trendMaxOrder=0, 
            no changepoints are allowed in the trend component; then a global polynomial model is used to fit the trend.
        tseg_minlength: int
            the min separation time between two neighboring trend changepoints.
        prior$precValue: numeric 
            the default value is 10.
        precPriorType: character 
            It takes one of 'constant', 'uniform' (the default), 'componentwise', and 'orderwise'. Below are the differences between them.
            - precPriorType='constant': the precision parameter used to parameterize the model coefficents is fixed to a constant specified by 
              precValue. In other words, precValue is a user-defined hyperparameter and the fitting result may be sensitive to the chosen 
              values of precValue.
            - precPriorType='uniform': the precision parameter used to parameterize the model coefficients is a random variable; 
              its initial value is specified by precValue. In other words, precValue will be infered by the MCMC, so the fitting 
              result is insenstive to the choice in precValue.
            - precPriorType='componentwise': multiple precision parameters are used to parameterize the model coefficients for 
              individual components (e.g., one for season and another for trend); their inital values is specified by prior$precValue.
              In other words, precValue will be infered by the MCMC, so the fitting result is insensitive to the choice in precValue.
            - precPriorType='orderwise': multiple precision parameters are used to parameterize the model coefficients not just for 
              individual components but also for individual orders of each component; their inital values is specified by prior$precValue.
              In other words, precValue will be inferred by the MCMC, so the fitting result is insensitive to the choice in precValue.
        
        
        # MCMC Paramaters
        --------------------------------------------

        Parameters to configure the MCMC inference. These parameter are not part of the Bayesian formulation of the BEAST model but are the settings
        for the reversible-jump MCMC to generate MCMC chains.Due to the MCMC nature, the longer the simulation chain is, the better the fitting result. 
        Below are possible parameters:
        
        seed: int (>=0) 
            the seed for the random number generator. If mcmcSeed=0, an arbitrary seed will be picked up and the fitting result will 
            var across runs. If fixed to the same on-zero integer, the results can be re-produced for different runs. Note that the 
            results may still vary if run on different computers with the same seed because the random generator libray depends on CPU's instruction sets.
        samples: int (>0) 
            the number of samples collected per MCMC chain.
        chainNumber: int (>0) 
            the number of parallel MCMC chains.
        thinningFactor: int (>0)
            a factor to thin chains (e.g., if thinningFactor=5, samples will be taken every 3 iterations).
        burnin: int (>0) 
            the number of burn-in samples discarded at the start of each chain.
        maxMoveStepSize: int (>0)
            The RJMCMC sampler employs a move proposal when traverseing the model space or proposing new positions of changepoints. 
            'maxMoveStepSize' is used in the 
            move proposal to specify the max window allowed in jumping from the current changepoint.
        seasonResamplingOrderProb: 
            a fractional number less than 1.0; the probability of selecting a re-sampling proposal 
            (e.g., resample seasonal harmonic order).
        trendResamplingOrderProb: 
            a fractional number less than 1.0; the probability of selecting a re-sampling proposal 
            (e.g., resample trend polynomial order)
        credIntervalAlphaLevel:
            a fractional number less than 1.0 (default to 0.95); the level of confidence used to compute credible intervals.
            

        # Extra Paramaters
        --------------------------------------------

        Flags to control the outputs from the BEAST runs or configure other program setting. Below are possible parameters:
        
        extra$dumpInputData: bool
            If TRUE, the input time series will be copied into the output. When the input Y is irregular (i.e., isRegularOrdered=FALSE), 
            the dumped copies will be the aggragated regular time series.
        whichOutputDimIsTime: int (<=3). 
            If the input Y is a 2D or 3D array (i.e., multiple time series such as stacked images), the whichOutputDimIsTime specifies which 
            dimension is the time in the output variables. whichOutputDimIsTime defaults to 3 for 3D inputs and is ignored if the input is 
            a vector (i.e., a single time series)
        computeCredible: logical (default to TRUE). 
            Credible intervals will be computed and outputted only if set to TRUE.
        fastCIComputation: logical (default to TRUE).
            If TRUE, a fast method is used to compute credible intervals (CI). Computation of CI is one of the most computational parts and 
            fastCIComputation should be set to TRUE unless more accurate CI estimation is desired.
        computeSeasonOrder: bool
            If TRUE, a posterior estimate of the seasonal harnomic order will be outputted; this flag is only valid if the time series has 
            a seasonal component (i.e., season='harmonic' and prior$seasonMinOrder is not equal to prior$seasonMaxOrder).
        computeTrendOrder: bool
            If TRUE, a posterior estimate of the tend polynomia order will be outputted; this flag is only valid when prior$trendMinOrder 
            is not equal to prior$trendMaxOrder).
        computeTrendOrder: bool
            TRUE, a posterior estimate of the tend polynomia order will be outputted; this flag is only valid when prior$trendMinOrder 
            is not equal to prior$trendMaxOrder).
        computeSeasonChngpt: bool
            If TRUE, compute the most likely times/positions where changepoints occur in the seasonal component. This flag is not 
            valid if there is a seasonal component in the time series (i.e., season='harmonic' or season='dummy' and prior$seasonMaxKnotNum is non-zero).
        computeTrendChngpt: bool
            If TRUE, compute the most likely times/positions where changepoints occur in the trend component.
        computeSeasonAmp: bool
            If TRUE, compute and output the time-varying amplitude of the seasonality.
        computeTrendSlope: bool
            If TRUE, compute and output the time-varying slope of the estimated trend.
        tallyPosNegSeasonJump: bool
            If TRUE, compute and differentite seasonal changepoints in terms of the direction of the jumps in the estimated seasonal 
            signal. Those changepoints with a positive jump will be outputted separately from those with a negative jump. A series of 
            output variables (some for postive-jump changepoints, and others for negative-jump changepoints will be dumped).
        tallyPosNegTrendJump:bool
            If TRUE, compute and differentite trend changepoints in terms of the direction of the jumps in the estimated trend. 
            Those changepoints with a positive jump will be outputted separately from those with a negative jump. A series of output 
            variables (some for postive-jump changepoints, and others for negative-jump changepoints will be dumped).
        tallyIncDecTrendJump: bool
            If TRUE, compute and differentite trend changepoints in terms of the direction of the jumps in the estimated slope of the trend signal. 
            Those changepoints with a increase in the slope will be outputted separately from those with a decrease in the slope. A series of output 
            variables (some for increase-jump changepoints, and others for decrease-jump changepoints will be dumped).
        printProgressBar: bool
            If TRUE, a progress bar will be displayed to show the status of the running. When running on multiple time series (e.g. stacked image time series), 
            the progress bar will also report an estimate of the remaining time for completion.
        consoleWidth: int bool
            the length of chars in each status line when setting printProgressBar=TRUE. If 0, the current width of the console will be used.
        printOptions: bool
            If TRUE, the values used in the arguments metadata, prior, mcmc, and extra will be printed to the console at the start of the run.
        numThreadsPerCPU: int 
            the number of threads to be scheduled for each CPU core.
        numParThreads: int 
            When handling many time series, BEAST can use multiple concurrent threads. extra$numParThreads specifies how many concurrent threads 
            will be used in total. If numParThreads=0, the actual number of threads will be numThreadsPerCPU * cpuCoreNumber; that is, each CPU core 
            will generate a number 'numThreadsPerCPU' of threads. On Windows 64, ,BEAST is group-aware and will affine the threads to all the NUMA node.
            But currently, up to 256 CPU cores are supported.
        
        ########################################################################
        # Result
        #########################################################################
        
        beastmaster.time	
            a vector of size 1xN: the times at the N sampled locatons. By default, it is simply set to 1:N
        beastmaster.data	
            a vector, matrix, or 3D array; this is a copy of the input Y if extra$dumpInputData = TRUE. If dumpInputData=FALSE, 
            it is set to NULL. If the orignal input Y is irregular, the copy here is the regular version aggragted from the original 
            at the time interval specified by metadata$deltaTime.
        beastmaster.marg_lik	
            numeric; the average of the model marginal likhood; the larger marg_lik, the better the fitting for a given time series.
        beastmaster.R2	
            numeric; the R-square of the model fiting.
        beastmaster.RMSE	
            numeric; the RMSE of the model fiting.
        beastmaster.sig2	
            numeric; the estimated variance of the model error.

        beastmaster.trend	
            an object consisting of various outputs related to the estimated trend component:
                - ncp: [Number of ChangePoints]. a numeric scalar; the mean number of trend changepoints. Individual models sampled by BEAST 
                        has a varying dimension (e.g., number of changeponts or knots), so several alternative statistics (e.g., ncp_mode, ncp_median, 
                        and ncp_pct90) are also given to summariize the number of changepoints. For example, if mcmc$samples=10, the 
                        numbers of changepoints for the 10 sampled models are assumed to be c(0, 2, 4, 1, 1, 2, 7, 6, 6, 1). The mean ncp is 3.1 
                        (rounded to 3), the median is 2.5 (2), the mode is 1, and the 90th percetile (ncp_pct90) is 6.5.
                - ncp_mode: [Number of ChangePoints]. a numeric scalar; the mode for number of changepoints. See the above for explanatatons.
                - ncp_median: [Number of ChangePoints]. a numeric scalar; the median for number of changepoints. See the above for explanatatons.
                - ncp_pct90: [Number of ChangePoints]. a numeric scalar; the 90th percetile for number of changepoints. See the above for explanatatons.
                - ncpPr: [Probability of the Number of ChangePoints]. A vector of length (prior$trendMaxKnotNum+1). It gives a probability distribution of
                        having a certain number of trend changepoints over the range of [0,prior$trendMaxKnotNum]; for example, ncpPr[1] is the probability
                        of having no trend changepoint; ncpPr[i] is the probability of having (i-1) changepoints: Note that it is ncpPr[i] not ncpPr[i-1] because 
                        ncpPr[1] is used for having zero changepoint.
                - cpOccPr: [ChangePoint OCCurence PRobability]. a vector of length N; it gives a probability distribution of having a changepoint in the trend at 
                        each point of time. Plotting cpOccPr will depict a continous curve of probability-of-being-changepoint. Of particular note, in the curve, a 
                        higher peak indicates a higher chance of being a changepoint only at that particular SINGLE point in time and does not neccessarily mean a higher 
                        chance of observing a changepoint AROUND that time. For example, a window of cpOccPr values c(0,0,0.5,0,0) (i.e., the peak prob is 0.5 and the summed 
                        prob is 0.5) is less likely to be a changepoint compared to another window c(0.1,0.2,0.21,0.2,0.1) (i.e., the peak prob is 0.21 but the summed prob is 0.71).
                - order: a vector of length N; the average polynomial order needed to approximate the fitted trend. As an average over many sampled individual piece-wise 
                        polynomial trends, order is not neccessarly an integer.
                - cp: [Changepoints] a vector of length tcp.max=tcp.minmax[2]; the most possible changepoint locations in the trend component. The locations are obtained by first 
                      applying a sum-filtering to the cpOccPr curve with a filter window size of tseg.min and then picking up to a total prior$MaxKnotNum/tcp.max of the highest 
                      peaks in the filterd curve. NaNs are possible if no enough changepoints are identified. cp records all the possible changepoints identified and many of them are 
                      bound to be false postives. Do not bindely treat all of them as actual changepoints.
                - cpPr: [Changepoints PRobability] a vector of length metadata$trendMaxKnotNum; the probabilities associated with the changepoints cp. Filled with NaNs 
                      for the remaning elements if ncp<trendMaxKnotNum.
                - cpCI: [Changepoints Credible Interval] a matrix of dimension metadata$trendMaxKnotNum x 2; the credibable intervals for the detected changepoints cp.
                - cpAbruptChange: [Abrupt change at Changepoints] a vector of length metadata$trendMaxKnotNum; the jumps in the fitted trend curves at the detected changepoints cp.
                - Y: a vector of length N; the estimated trend component. It is the Bayesian model averaging of all the individual sampled trend.
                - SD: [Standard Deviation] a vector of length N; the estimated standard deviation of the estimated trend component.

                
        beastmaster.season	
            a list object numeric consisting of various outputs related to the estimated seasonal/periodic component:
            - ncp:   [Number of ChangePoints]. a numeric scalar; the mean number of seasonal changepoints.
            - ncpPr: [Probability of the Number of ChangePoints]. A vector of length (prior$seasonMaxKnotNum+1). It gives a probability distribution of having a 
                     certain number of seasonal changepoints over the range of [0,prior$seasonMaxKnotNum]; for example, ncpPr[1] is the probability of having no 
                     seasonal changepoint; ncpPr[i] is the probability of having (i-1) changepoints: Note that the index is i rather than (i-1) because ncpPr[1] is used for having zero changepoint.
            - cpOccPr: [ChangePoint OCCurence PRobability]. a vector of length N; it gives a probability distribution of having a changepoint in the seasonal component at each point of time. Plotting 
                      cpOccPr will depict a continous curve of probability-of-being-changepoint over the time. Of particular note, in the curve, a higher value at a peak indicates a higher chance of being a changepoint only at that particular 
                      SINGLE point in time, and does not neccessarily mean a higher chance of observing a changepoint AROUND that time. For example, a window of cpOccPr values c(0,0,0.5,0,0) (i.e., the peak prob is 0.5 and the 
                      summed prob is 0.5) is less likely to be a changepoint compared to another window values c(0.1,0.2,0.3,0.2,0.1) (i.e., the peak prob is 0.3 but the summed prob is 0.8).
            - order: a vector of length N; the average harmonic order needed to approximate the seasonal component. As an average over many sampled individual piece-wise harmonic curves, order is not neccessarly an integer.
            - cp: [Changepoints] a vector of length metadata$seasonMaxKnotNum; the most possible changepoint locations in the seasonal component. The locations are obtained by first applying a sum-filtering to the 
                 cpOccPr curve with a filter window size of prior$trendMinSeptDist and then picking up to a total ncp of the highest peaks in the filterd curve. If ncp<seasonMaxKnotNum, the remaining of the vector is filled with NaNs.
            - cpPr: [Changepoints PRobability] a vector of length metadata$seasonMaxKnotNum; the probabilities associated with the changepoints cp. Filled with NaNs for the remaning elements if ncp<seasonMaxKnotNum.
            - cpCI: [Changepoints Credible Interval] a matrix of dimension metadata$seasonMaxKnotNum x 2; the credibable intervals for the detected changepoints cp.
            - cpAbruptChange: [Abrupt change at Changepoints] a vector of length metadata$seasonMaxKnotNum; the jumps in the fitted trend curves at the detected changepoints cp.
            - Y: a vector of length N; the estimated trend component. It is the Bayesian model averaging of all the individual sampled trend.
            - SD: [Standard Deviation] a vector of length N; the estimated standard deviation of the estimated trend component.

        """
        
        ## init metadata
        self.metadata = rb.args()
    
        self.metadata.time = time
        self.metadata.isRegularOrdered = isRegularOrdered
        self.metadata.season = season
        self.metadata.startTime = startTime
        self.metadata.deltaTime = deltaTime
        self.metadata.freq = freq
        if (self.season != 'none'):
            self.metadata.period = self.metadata.deltaTime * self.metadata.freq
        self.metadata.missingValue = missingValue
        self.metadata.maxMissingRate = maxMissingRate
        self.metadata.freq = freq
        self.metadata.deseasonalize = deseasonalize
        self.metadata.detrend = detrend
        self.metadata.hasOutlierCmpnt  = hasOutlier
        self.metadata.whichDimIsTime = whichDimIsTime

   
        #init prior
        self.prior = rb.args()
        self.prior.modelPriorType	  = 1
        if season !='none' or season == None:
            self.prior.seasonMinOrder   = sorder_minmax[0]
            self.prior.seasonMaxOrder   = sorder_minmax[1]
            self.prior.seasonMinKnotNum = scp_minmax[0]
            self.prior.seasonMaxKnotNum = scp_minmax[1]
            self.prior.seasonMinSepDist = sseg_minlength
        self.prior.trendMinOrder	  = torder_minmax[0]
        self.prior.trendMaxOrder	  = torder_minmax[1]
        self.prior.trendMinKnotNum  = tcp_minmax[0]
        self.prior.trendMaxKnotNum  = tcp_minmax[1]
        self.prior.trendMinSepDist  = tseg_minlength
        self.prior.K_MAX            = 500
        self.prior.precValue = precValue
        self.prior.precPriorType = precPriorType
        
        # init mcmc
        self.mcmc = rb.args()
        self.mcmc.seed = seed
        self.mcmc.samples = samples
        self.mcmc.thinningFactor = thinningFactor
        self.mcmc.burnin = burnin
        self.mcmc.chainNumber = chainNumber
        self.mcmc.maxMoveStepSize = maxMoveStepSize
        self.mcmc.trendResamplingOrderProb = trendResamplingOrderProb
        self.mcmc.seasonResamplingOrderProb = seasonResamplingOrderProb
        self.mcmc.credIntervalAlphaLevel = credIntervalAlphaLevel 
        
        #init extra
        self.extra = rb.args()
        self.extra.dumpInputData = dumpInputData
        self.extra.whichOutputDimIsTime = whichOutputDimIsTime
        self.extra.computeCredible = computeCredible
        self.extra.fastCIComputation = fastCIComputation
        self.extra.computeTrendOrder = computeTrendOrder
        self.extra.computeTrendChngpt = computeTrendChngpt
        self.extra.computeTrendSlope = computeTrendSlope
        self.extra.computeSeasonOrder = computeSeasonOrder
        self.extra.computeSeasonChngpt = computeSeasonChngpt
        self.extra.computeSeasonAmp = computeSeasonAmp
        self.extra.tallyPosNegTrendJump = tallyPosNegTrendJump
        self.extra.tallyIncDecTrendJump = tallyIncDecTrendJump 
        self.extra.printProgressBar = printProgressBar
        self.extra.printOptions = printOptions
        self.extra.consoleWidth = consoleWidth
        self.extra.numThreadsPerCPU = numThreadsPerCPU
        self.extra.numParThreads = numParThreads
        
    def beastrunner(self, dataset):
        """
        dataset: array
            a 1D array, 2D array, or 3D array of numeric data. Missing values are allowed and can be indicated 
            by NA, NaN, or a value customized in the by the missingValue paramater.
            - If dataset is a vector of size Nx1 or 1xN, it is treated as a single time series of length N.
            - If dataset is a 2D matrix or 3D array of dimension N1xN2 or N1xN2xN3 (e.g., stacked images of geospatial data), 
            it includes multiple time series of equal length   
        """
        self.dataset = dataset
        self.ndims = self.dataset.ndim
        if self.ndims == 1 and self.isRegularOrdered == True:
            self.out = rb.beast(self.dataset, self.metadata, self.prior, self.mcmc, self.extra)
        elif self.ndims == 1 and self.isRegularOrdered == False:
            self.out = rb.beast_irreg(self.dataset, self.metadata, self.prior, self.mcmc, self.extra)
        elif self.ndims >= 1:
            self.out = rb.beast123(self.dataset, self.metadata, self.prior, self.mcmc, self.extra)

        self.time = self.out.time
        self.marg_lik = self.out.marg_lik
        self.R2 = self.out.R2
        self.RMSE = self.out.RMSE
        self.sig2 = self.out.sig2
        self.trend = self.out.trend
        self.ncp = self.trend.ncp
        self.ncp_median = self.trend.ncp_median
        self.ncp_mode = self.trend.ncp_mode
        self.ncp_pct90 = self.trend.ncp_pct90
        self.ncp_pct10 = self.trend.ncp_pct10
        self.cpOccPr = self.trend.cpOccPr
        self.cp = self.trend.cp
        self.cpPr = self.trend.cpPr
        self.cpAbruptChange = self.trend.cpAbruptChang
        self.cpCI = self.trend.cpCI
        self.Y = self.trend.Y
        self.SD = self.trend.SD

        if self.season !='none' or self.eason == None:
            self.season = self.season
            self.ncp = self.season.ncp
            self.ncp_median = self.season.ncp_median
            self.ncp_mode = self.season.ncp_mode
            self.ncp_pct90 = self.season.ncp_pct90
            self.ncp_pct10 = self.season.ncp_pct10
            self.ncpPr = self.season.ncpPr
            self.cpOccPr = self.season.cpOccPr
            self.cp = self.season.cp
            self.cpPr = self.season.cpPr
            self.cpAbruptChange = self.season.cpAbruptChange
            self.cpCIr = self.season.cpCI
            self.Y = self.season.Y
            self.SD = self.season.SD
        self.whichOutDimIsTime = self.out.whichOutDimIsTime
        self.nrows = self.out.nrows
        self.ncols = self.out.ncols
        self.season_type = self.out.season_type
    
    def get_result(self):
        return self.out
    
    def plot(self):
        rb.plot(o[10,11])

    def print_params(self):
        if len(self.metadata.__dict__) >= 0:
            print('Metadata Parameters')
            print('--------------------------------------------------------')
            meta_table = [['Parameter', 'Value']]
            for key in self.metadata.__dict__.keys():
                if key == 'time':
                    print(f'time_steps : {len(self.metadata.__dict__[key])}')
                else:
                    print(f'{key} : {self.metadata.__dict__[key]}')
            print('--------------------------------------------------------')
        print('Hyperprior Parameters')
        print('--------------------------------------------------------')
        meta_table = [['Parameter', 'Value']]
        for key in self.prior.__dict__.keys():
            print(f'{key} : {self.prior.__dict__[key]}')
        print('--------------------------------------------------------')

        print('MCMC Parameters')
        print('--------------------------------------------------------')
        meta_table = [['Parameter', 'Value']]
        for key in self.mcmc.__dict__.keys():
            print(f'{key} : {self.mcmc.__dict__[key]}')
        print('--------------------------------------------------------')
    
        print('Extra Parameters')
        print('--------------------------------------------------------')
        meta_table = [['Parameter', 'Value']]
        for key in self.extra.__dict__.keys():
            print(f'{key} : {self.extra.__dict__[key]}')
        print('--------------------------------------------------------')


