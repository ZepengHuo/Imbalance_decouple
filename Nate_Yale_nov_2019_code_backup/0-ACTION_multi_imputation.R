setwd("/data/Projects/ACC_NCDR/NCDR/BJMDATA/ACTION/")
load(file='/data/Projects/ACC_NCDR/NCDR/BJMDATA/ACTION/RData/patientmerged_meds.RData')

library(mice)

#Exclude patients transferred to another acute care facility
cohort.data <- cohort.data[-which(cohort.data$DCLocation == 3) ,]

#Dates for cohort selection, starts at 2011 to account for CAFirstMedCon coding
cohort.data$DATE <- as.Date(as.POSIXct(cohort.data$Timeframe, origin = "1960-01-01"))
start.date <- as.Date(as.POSIXct("2011-01-01"))
end.date <- as.Date(as.POSIXct("2017-01-01"))
timeinterval.data <- cohort.data[which(cohort.data$DATE > start.date & cohort.data$DATE <= end.date) ,]


race.categorical.variables <- c('RaceWhite',
                                'Sex',
                                'RaceBlack',
                                'RaceAsian',
                                'RaceAmIndian',
                                'RaceNatHaw',
                                'HispOrig',
                                'DATE')

lab.categorical.variables <- c('InitTrop',
                               'InitCKMB',
                               'InitCreat',
                               'InitHGB',
                               'Lipids',
                               'BNP',
                               'proBNP',
                               'PosMarkers')

cardiac.categorical.variables <- c('ECGFindings',
                                   'StemiNoted',
                                   'OtherECGFindings',
                                   'HFFirstMedCon',
                                   'ShockFirstMedCon',
                                   #'Cocaine',
                                   'CAFirstMedCon')

history.categorical.variables <- c('Hypertension',
                                   'Smoker',
                                   'Dyslipidemia',
                                   'CurrentDialysis',
                                   'ChronicLungDisease', #Note- comment out for Julian's model,  keep in for McNamara
                                   'Diabetes',
                                   'DiabetesControl',
                                   'PriorMI',
                                   'PriorHF',
                                   'PriorPCI',
                                   'PriorCABG',
                                   'PriorAfib',
                                   'PriorCVD',
                                   'PriorStroke',
                                   'PriorTIA',
                                   'PriorPAD'
                                   )

inhosp.categorical.variables <- c('CE_CardioShock',
                                  'CE_HF',
                                  'CE_CardiacArrest',
                                  'CE_Bleeding',
                                  'CE_RBC')

meds.home.categorical.variables <- names(cohort.data)[which(grepl("Home", names(cohort.data)))]



all.categorical.variables <- c(race.categorical.variables, 
                               lab.categorical.variables,
                               cardiac.categorical.variables,
                               history.categorical.variables, 
                              # inhosp.categorical.variables,
                               meds.home.categorical.variables
)

continuous.variables <- c('InitTropValue',
                              'InitTropURL',
                              #'InitCKMBValue',
                              #'InitCKMBULN',
                              'InitCreatValue',
                              'InitHGBValue',
                              #'InitINRValue',
                              #'LipidsTC',
                              #'LipidsHDL',
                              #'LipidsLDL',
                              #'LipidsTrig',
                          'Age',
                          'Weight',
                          'Height',
                          'HRFirstMedCon',
                          'SBPFirstMedCon'
)

sample.data <- timeinterval.data[, c(all.categorical.variables, continuous.variables, "DCStatus")]
sample.data <- sample.data[which(!is.na(sample.data$PriorPCI)) ,]

sample.data$DCStatus[which(sample.data$DCStatus == 1)] <- 0
sample.data$DCStatus[which(sample.data$DCStatus == 2)] <- 1

sample.data$InitTrop[which(sample.data$InitTrop == 2)] <- 1

sample.data$PriorStroke[which(sample.data$PriorCVD == 0)] <- 0
sample.data$PriorTIA[which(sample.data$PriorCVD == 0)] <- 0

#Divide STEMI variable into separate subclasses based on ECGFindings variable
sample.data$Stemi_ST <- 0
sample.data$Stemi_LBBB <- 0
sample.data$Stemi_PostMI <- 0

sample.data$Stemi_ST[which(sample.data$ECGFindings == 1)] <- 1
sample.data$Stemi_LBBB[which(sample.data$ECGFindings == 2)] <- 1
sample.data$Stemi_PostMI[which(sample.data$ECGFindings == 3)] <- 1

#Divide non-stemi variable into separate subclasses based on OtherECGFindings variable
sample.data$NoStemi_STdep <- 0
sample.data$NoStemi_Twave <- 0
sample.data$NoStemi_transST <- 0
sample.data$NoStemi_none <- 0
sample.data$NoStemi_oldLBBB <- 0
sample.data$NoStemi_other <- 0 

sample.data$NoStemi_STdep[which(sample.data$OtherECGFindings == 1)] <- 1
sample.data$NoStemi_Twave[which(sample.data$OtherECGFindings == 2)] <- 1
sample.data$NoStemi_transST[which(sample.data$OtherECGFindings == 3)] <- 1
sample.data$NoStemi_none[which(sample.data$OtherECGFindings == 4)] <- 1
sample.data$NoStemi_none[which(is.na(sample.data$OtherECGFindings))] <- 1
sample.data$NoStemi_oldLBBB[which(sample.data$OtherECGFindings == 5)] <- 1
sample.data$NoStemi_other[which(sample.data$OtherECGFindings == 6)] <- 1

#Divide diabetes into separate treatment groups diabetes control
sample.data$DM_notx <- 0
sample.data$DM_diet <- 0
sample.data$DM_oral <- 0
sample.data$DM_insulin <- 0 
sample.data$DM_other <- 0

sample.data$DM_notx[which(sample.data$DiabetesControl == 1)] <- 1
sample.data$DM_diet[which(sample.data$DiabetesControl == 2)] <- 1
sample.data$DM_oral[which(sample.data$DiabetesControl == 3)] <- 1
sample.data$DM_insulin[which(sample.data$DiabetesControl == 4)] <- 1
sample.data$DM_other[which(sample.data$DiabetesControl == 5)] <- 1

ecganddm.categorical.variables <- c('Stemi_ST',
                                    'Stemi_LBBB',
                                    'Stemi_PostMI',
                                    'NoStemi_STdep',
                                    'NoStemi_Twave',
                                    'NoStemi_transST',
                                    'NoStemi_none',
                                    'NoStemi_oldLBBB',
                                    'NoStemi_other',
                                    'DM_notx',
                                    'DM_diet',
                                    'DM_oral',
                                    'DM_insulin',
                                    'DM_other')

all.categorical.variables <- c(race.categorical.variables, 
                               lab.categorical.variables,
                               cardiac.categorical.variables,
                               history.categorical.variables, 
                              # inhosp.categorical.variables,
                               meds.home.categorical.variables,
                               ecganddm.categorical.variables)
                               

#remove meds missing for majority of records
#NCH addition 5/1/2019: Remove meds as per Rohan/Harlan
drops <- c(
    "Dabigatran_Home",
    "Rivaroxaban_Home",
    "Apixaban_Home",
    "Tic_Home",
    "ECGFindings",
    "OtherECGFindings",
    "DiabetesControl",
    "DATE",
    "PriorTIA",
    "ASA_Home",
    "CLPD_Home",
    "Ticlid_Home",
    "ARB_Home",
    "BB_Home",
    "Statin_Home",
    "OLLA_Home",
    "Prasu_Home",
    "War_Home",
    "ABA_Home"
          )
sample.data <- sample.data[, !(names(sample.data) %in% drops)]
all.categorical.variables <- all.categorical.variables[! all.categorical.variables %in% drops]


labels <- sample.data$DCStatus

sample.data$DCStatus <- NULL

for (var in all.categorical.variables){
  var.col <- sample.data[, var]
  if(any(is.na(var.col))){
    #mode.val <- Mode(var.col[which(!is.na(var.col))])
    imputed <- which(is.na(var.col))
    #sample.data[imputed, var] <- mode.val
    sample.data$imp <- 0
    sample.data$imp[imputed] <- 1
    names(sample.data)[which(names(sample.data) == 'imp')] <- paste(var, '_IMP', sep ='')
  }
}

for (var in continuous.variables){
  var.col <- sample.data[, var]
  if(any(is.na(var.col))){
    #median.val <- median(var.col[which(!is.na(var.col))])
    imputed <- which(is.na(var.col))
    #sample.data[imputed, var] <- median.val
    sample.data$imp <- 0
    sample.data$imp[imputed] <- 1
    names(sample.data)[which(names(sample.data) == 'imp')] <- paste(var, '_IMP', sep ='')
  }
}

print(Sys.time())
ptm <- proc.time()
imp <- mice(sample.data, printFlag=FALSE)
proc.time() - ptm
print(Sys.time())

fold.1 <- complete(imp, 1)
fold.2 <- complete(imp, 2)
fold.3 <- complete(imp, 3)
fold.4 <- complete(imp, 4)
fold.5 <- complete(imp, 5)

fold.1$InitTropValue[which(fold.1$InitTropValue >= 2)] <- 2
fold.1$InitTropURL[which(fold.1$InitTropURL >= 0.2)] <- 0.2

#fold.1$InitCKMBValue[which(fold.1$InitCKMBValue >= 40)] <- 40 
#fold.1$InitCKMBULN[which(fold.1$InitCKMBULN >= 25)] <- 25

fold.1$InitCreatValue[which(fold.1$InitCreatValue >= 15)] <- 15
fold.1$InitHGBValue[which(fold.1$InitHGBValue >= 20)] <- 17
#fold.1$InitINRValue[which(fold.1$InitINRValue >= 10)] <- 7

#fold.1$LipidsTC[which(fold.1$LipidsTC >= 350)] <- 350
#fold.1$LipidsHDL[which(fold.1$LipidsHDL >= 100)] <- 100
#fold.1$LipidsLDL[which(fold.1$LipidsLDL >= 300)] <- 300
#fold.1$LipidsTrig[which(fold.1$LipidsTrig >= 1200)] <- 1200

fold.1$Age[which(fold.1$Age >= 100)] <- 100
fold.1$Weight[which(fold.1$Weight >= 225)] <- 225
fold.1$Height[which(fold.1$Height >= 200)] <- 215


#BMI, max = 60 kg/m^2
fold.1$BMI <- fold.1$Weight / ((fold.1$Height/100)^2)
fold.1$BMI[which(fold.1$BMI < 15)] <- 15
fold.1$BMI[which(fold.1$BMI > 60)] <- 60 

#CrCl
fold.1$CrCl <- 0
fold.1$CrCl[which(fold.1$Sex == 2)] <- (   (140-fold.1$Age[which(fold.1$Sex == 2)]) * fold.1$Weight[which(fold.1$Sex == 2)]  ) / (fold.1$InitCreatValue[which(fold.1$Sex == 2)] * 72) * 0.85  
fold.1$CrCl[which(fold.1$Sex == 1)] <- (   (140-fold.1$Age[which(fold.1$Sex == 1)]) * fold.1$Weight[which(fold.1$Sex == 1)]  ) / (fold.1$InitCreatValue[which(fold.1$Sex == 1)] * 72)
fold.1$CrCl[which(fold.1$CrCl >= 200)] <- 200

#calculate troponin ratio, max 20
fold.1$TropRatio <- fold.1$InitTropValue / fold.1$InitTropURL
fold.1$TropRatio[which(fold.1$TropRatio >= 20)] <- 20 
fold.1$TropRatio[which(is.na(fold.1$TropRatio))] <- 0

#calculate CKMB ratio, max 2
# fold.1$CKMBRatio <- fold.1$InitCKMBValue / fold.1$InitCKMBULN
# fold.1$CKMBRatio[which(fold.1$CKMBRatio >= 2)] <- 2
# fold.1$CKMBRatio[which(is.na(fold.1$CKMBRatio))] <- 0
#save(fold.1, file='/data/Projects/ACC_NCDR/NCDR/BJMDATA/ACTION/NCHData/multiple_imputed/modeldata_imputed_2011_fold.1.RData')

fold.2$InitTropValue[which(fold.2$InitTropValue >= 2)] <- 2
fold.2$InitTropURL[which(fold.2$InitTropURL >= 0.2)] <- 0.2

#fold.2$InitCKMBValue[which(fold.2$InitCKMBValue >= 40)] <- 40 
#fold.2$InitCKMBULN[which(fold.2$InitCKMBULN >= 25)] <- 25

fold.2$InitCreatValue[which(fold.2$InitCreatValue >= 15)] <- 15
fold.2$InitHGBValue[which(fold.2$InitHGBValue >= 20)] <- 17
#fold.2$InitINRValue[which(fold.2$InitINRValue >= 10)] <- 7

#fold.2$LipidsTC[which(fold.2$LipidsTC >= 350)] <- 350
#fold.2$LipidsHDL[which(fold.2$LipidsHDL >= 100)] <- 100
#fold.2$LipidsLDL[which(fold.2$LipidsLDL >= 300)] <- 300
#fold.2$LipidsTrig[which(fold.2$LipidsTrig >= 1200)] <- 1200

fold.2$Age[which(fold.2$Age >= 100)] <- 100
fold.2$Weight[which(fold.2$Weight >= 225)] <- 225
fold.2$Height[which(fold.2$Height >= 200)] <- 215


#BMI, max = 60 kg/m^2
fold.2$BMI <- fold.2$Weight / ((fold.2$Height/100)^2)
fold.2$BMI[which(fold.2$BMI < 15)] <- 15
fold.2$BMI[which(fold.2$BMI > 60)] <- 60 

#CrCl
fold.2$CrCl <- 0
fold.2$CrCl[which(fold.2$Sex == 2)] <- (   (140-fold.2$Age[which(fold.2$Sex == 2)]) * fold.2$Weight[which(fold.2$Sex == 2)]  ) / (fold.2$InitCreatValue[which(fold.2$Sex == 2)] * 72) * 0.85  
fold.2$CrCl[which(fold.2$Sex == 1)] <- (   (140-fold.2$Age[which(fold.2$Sex == 1)]) * fold.2$Weight[which(fold.2$Sex == 1)]  ) / (fold.2$InitCreatValue[which(fold.2$Sex == 1)] * 72)
fold.2$CrCl[which(fold.2$CrCl >= 200)] <- 200

#calculate troponin ratio, max 20
fold.2$TropRatio <- fold.2$InitTropValue / fold.2$InitTropURL
fold.2$TropRatio[which(fold.2$TropRatio >= 20)] <- 20 
fold.2$TropRatio[which(is.na(fold.2$TropRatio))] <- 0

#calculate CKMB ratio, max 2
# fold.2$CKMBRatio <- fold.2$InitCKMBValue / fold.2$InitCKMBULN
# fold.2$CKMBRatio[which(fold.2$CKMBRatio >= 2)] <- 2
# fold.2$CKMBRatio[which(is.na(fold.2$CKMBRatio))] <- 0
#save(fold.2, file='/data/Projects/ACC_NCDR/NCDR/BJMDATA/ACTION/NCHData/multiple_imputed/modeldata_imputed_2011_fold.2.RData')

fold.3$InitTropValue[which(fold.3$InitTropValue >= 2)] <- 2
fold.3$InitTropURL[which(fold.3$InitTropURL >= 0.2)] <- 0.2

#fold.3$InitCKMBValue[which(fold.3$InitCKMBValue >= 40)] <- 40 
#fold.3$InitCKMBULN[which(fold.3$InitCKMBULN >= 25)] <- 25

fold.3$InitCreatValue[which(fold.3$InitCreatValue >= 15)] <- 15
fold.3$InitHGBValue[which(fold.3$InitHGBValue >= 20)] <- 17
#fold.3$InitINRValue[which(fold.3$InitINRValue >= 10)] <- 7

#fold.3$LipidsTC[which(fold.3$LipidsTC >= 350)] <- 350
#fold.3$LipidsHDL[which(fold.3$LipidsHDL >= 100)] <- 100
#fold.3$LipidsLDL[which(fold.3$LipidsLDL >= 300)] <- 300
#fold.3$LipidsTrig[which(fold.3$LipidsTrig >= 1200)] <- 1200

fold.3$Age[which(fold.3$Age >= 100)] <- 100
fold.3$Weight[which(fold.3$Weight >= 225)] <- 225
fold.3$Height[which(fold.3$Height >= 200)] <- 215


#BMI, max = 60 kg/m^2
fold.3$BMI <- fold.3$Weight / ((fold.3$Height/100)^2)
fold.3$BMI[which(fold.3$BMI < 15)] <- 15
fold.3$BMI[which(fold.3$BMI > 60)] <- 60 

#CrCl
fold.3$CrCl <- 0
fold.3$CrCl[which(fold.3$Sex == 2)] <- (   (140-fold.3$Age[which(fold.3$Sex == 2)]) * fold.3$Weight[which(fold.3$Sex == 2)]  ) / (fold.3$InitCreatValue[which(fold.3$Sex == 2)] * 72) * 0.85  
fold.3$CrCl[which(fold.3$Sex == 1)] <- (   (140-fold.3$Age[which(fold.3$Sex == 1)]) * fold.3$Weight[which(fold.3$Sex == 1)]  ) / (fold.3$InitCreatValue[which(fold.3$Sex == 1)] * 72)
fold.3$CrCl[which(fold.3$CrCl >= 200)] <- 200

#calculate troponin ratio, max 20
fold.3$TropRatio <- fold.3$InitTropValue / fold.3$InitTropURL
fold.3$TropRatio[which(fold.3$TropRatio >= 20)] <- 20 
fold.3$TropRatio[which(is.na(fold.3$TropRatio))] <- 0

#calculate CKMB ratio, max 2
# fold.3$CKMBRatio <- fold.3$InitCKMBValue / fold.3$InitCKMBULN
# fold.3$CKMBRatio[which(fold.3$CKMBRatio >= 2)] <- 2
# fold.3$CKMBRatio[which(is.na(fold.3$CKMBRatio))] <- 0
#save(fold.3, file='/data/Projects/ACC_NCDR/NCDR/BJMDATA/ACTION/NCHData/multiple_imputed/modeldata_imputed_2011_fold.3.RData')

fold.4$InitTropValue[which(fold.4$InitTropValue >= 2)] <- 2
fold.4$InitTropURL[which(fold.4$InitTropURL >= 0.2)] <- 0.2

#fold.4$InitCKMBValue[which(fold.4$InitCKMBValue >= 40)] <- 40 
#fold.4$InitCKMBULN[which(fold.4$InitCKMBULN >= 25)] <- 25

fold.4$InitCreatValue[which(fold.4$InitCreatValue >= 15)] <- 15
fold.4$InitHGBValue[which(fold.4$InitHGBValue >= 20)] <- 17
#fold.4$InitINRValue[which(fold.4$InitINRValue >= 10)] <- 7

#fold.4$LipidsTC[which(fold.4$LipidsTC >= 350)] <- 350
#fold.4$LipidsHDL[which(fold.4$LipidsHDL >= 100)] <- 100
#fold.4$LipidsLDL[which(fold.4$LipidsLDL >= 300)] <- 300
#fold.4$LipidsTrig[which(fold.4$LipidsTrig >= 1200)] <- 1200

fold.4$Age[which(fold.4$Age >= 100)] <- 100
fold.4$Weight[which(fold.4$Weight >= 225)] <- 225
fold.4$Height[which(fold.4$Height >= 200)] <- 215


#BMI, max = 60 kg/m^2
fold.4$BMI <- fold.4$Weight / ((fold.4$Height/100)^2)
fold.4$BMI[which(fold.4$BMI < 15)] <- 15
fold.4$BMI[which(fold.4$BMI > 60)] <- 60 

#CrCl
fold.4$CrCl <- 0
fold.4$CrCl[which(fold.4$Sex == 2)] <- (   (140-fold.4$Age[which(fold.4$Sex == 2)]) * fold.4$Weight[which(fold.4$Sex == 2)]  ) / (fold.4$InitCreatValue[which(fold.4$Sex == 2)] * 72) * 0.85  
fold.4$CrCl[which(fold.4$Sex == 1)] <- (   (140-fold.4$Age[which(fold.4$Sex == 1)]) * fold.4$Weight[which(fold.4$Sex == 1)]  ) / (fold.4$InitCreatValue[which(fold.4$Sex == 1)] * 72)
fold.4$CrCl[which(fold.4$CrCl >= 200)] <- 200

#calculate troponin ratio, max 20
fold.4$TropRatio <- fold.4$InitTropValue / fold.4$InitTropURL
fold.4$TropRatio[which(fold.4$TropRatio >= 20)] <- 20 
fold.4$TropRatio[which(is.na(fold.4$TropRatio))] <- 0

#calculate CKMB ratio, max 2
# fold.4$CKMBRatio <- fold.4$InitCKMBValue / fold.4$InitCKMBULN
# fold.4$CKMBRatio[which(fold.4$CKMBRatio >= 2)] <- 2
# fold.4$CKMBRatio[which(is.na(fold.4$CKMBRatio))] <- 0
#save(fold.4, file='/data/Projects/ACC_NCDR/NCDR/BJMDATA/ACTION/NCHData/multiple_imputed/modeldata_imputed_2011_fold.4.RData')

fold.5$InitTropValue[which(fold.5$InitTropValue >= 2)] <- 2
fold.5$InitTropURL[which(fold.5$InitTropURL >= 0.2)] <- 0.2

#fold.5$InitCKMBValue[which(fold.5$InitCKMBValue >= 40)] <- 40 
#fold.5$InitCKMBULN[which(fold.5$InitCKMBULN >= 25)] <- 25

fold.5$InitCreatValue[which(fold.5$InitCreatValue >= 15)] <- 15
fold.5$InitHGBValue[which(fold.5$InitHGBValue >= 20)] <- 17
#fold.5$InitINRValue[which(fold.5$InitINRValue >= 10)] <- 7

#fold.5$LipidsTC[which(fold.5$LipidsTC >= 350)] <- 350
#fold.5$LipidsHDL[which(fold.5$LipidsHDL >= 100)] <- 100
#fold.5$LipidsLDL[which(fold.5$LipidsLDL >= 300)] <- 300
#fold.5$LipidsTrig[which(fold.5$LipidsTrig >= 1200)] <- 1200

fold.5$Age[which(fold.5$Age >= 100)] <- 100
fold.5$Weight[which(fold.5$Weight >= 225)] <- 225
fold.5$Height[which(fold.5$Height >= 200)] <- 215


#BMI, max = 60 kg/m^2
fold.5$BMI <- fold.5$Weight / ((fold.5$Height/100)^2)
fold.5$BMI[which(fold.5$BMI < 15)] <- 15
fold.5$BMI[which(fold.5$BMI > 60)] <- 60 

#CrCl
fold.5$CrCl <- 0
fold.5$CrCl[which(fold.5$Sex == 2)] <- (   (140-fold.5$Age[which(fold.5$Sex == 2)]) * fold.5$Weight[which(fold.5$Sex == 2)]  ) / (fold.5$InitCreatValue[which(fold.5$Sex == 2)] * 72) * 0.85  
fold.5$CrCl[which(fold.5$Sex == 1)] <- (   (140-fold.5$Age[which(fold.5$Sex == 1)]) * fold.5$Weight[which(fold.5$Sex == 1)]  ) / (fold.5$InitCreatValue[which(fold.5$Sex == 1)] * 72)
fold.5$CrCl[which(fold.5$CrCl >= 200)] <- 200

#calculate troponin ratio, max 20
fold.5$TropRatio <- fold.5$InitTropValue / fold.5$InitTropURL
fold.5$TropRatio[which(fold.5$TropRatio >= 20)] <- 20 
fold.5$TropRatio[which(is.na(fold.5$TropRatio))] <- 0

#calculate CKMB ratio, max 2
# fold.5$CKMBRatio <- fold.5$InitCKMBValue / fold.5$InitCKMBULN
# fold.5$CKMBRatio[which(fold.5$CKMBRatio >= 2)] <- 2
# fold.5$CKMBRatio[which(is.na(fold.5$CKMBRatio))] <- 0
#save(fold.5, file='/data/Projects/ACC_NCDR/NCDR/BJMDATA/ACTION/NCHData/multiple_imputed/modeldata_imputed_2011_fold.5.RData')

save(fold.1, fold.2, fold.3, fold.4, fold.5, labels, file='/data/Projects/ACC_NCDR/NCDR/BJMDATA/ACTION/NCHData/multiple_imputed/all_imputed_folds.RData')

