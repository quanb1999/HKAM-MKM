clear

load('Data_sets_PDB1075_186.mat');

train_X = [GE_1075,MCD_1075,NMBAC_1075,PSSM_AB_1075,PSSM_Pse_1075,PSSM_DWT_1075];
test_X = [GE_186,MCD_186,NMBAC_186,PSSM_AB_186,PSSM_Pse_186,PSSM_DWT_186];
COM_X = [train_X;test_X];
COM_X = line_map(COM_X);
train_X_S = COM_X(1:1075,:);
test_X_S = COM_X(1076:end,:);

  

feature_id=[1,150;151,1032;1033,1232;1233,1432;1433,1652;1653,2692];
gamma_list = [2^-1,2^-3,2^-1,2^-0,2^-1,2^-4];
% feature_id=[151,1032;1233,1432;1433,1652];
% gamma_list = [2^-3,2^-0,2^-1];
c =3;


k=35;
lamda=2^-5;
IsMK='MKL-HKA';
[predict_y,Scores,kernel_weights] = msvm_hka(train_X_S,feature_id,label_1075,test_X_S,label_186,c,gamma_list,k,lamda,IsMK);
[ACC,SN,Spec,PE,NPV,F_score,MCC,auc] = roc( predict_y,Scores,label_186 )

