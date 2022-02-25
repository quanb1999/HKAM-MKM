clear
load('Data_sets_PDB14189_2272.mat');

train_X = [GE_14189,MCD_14189,NMBAC_14189,PSSM_AB_14189,PSSM_Pse_14189,PSSM_DWT_14189];
test_X = [GE_2272,MCD_2272,NMBAC_2272,PSSM_AB_2272,PSSM_Pse_2272,PSSM_DWT_2272];


COM_X = [train_X;test_X];
COM_X = line_map(COM_X);
train_X_S = COM_X(1:14189,:);
test_X_S = COM_X(14190:end,:);


gamma_list = [2^-1,2^-3,2^-1,2^-0,2^-1,2^-4];
%gamma_list = [2^-1,2^-1,2^-4];
feature_id=[1,150;151,1032;1033,1232;1233,1432;1433,1652;1653,2692];
%feature_id=[1,150;1433,1652;1653,2692];

c =3;

k=10;
lamda=2^-3;
IsMK = 'MKL-HKA';
[predict_y,Scores,kernel_weights] = msvm_hka(train_X_S,feature_id,label_14189,test_X_S,label_2272,c,gamma_list,k,lamda,IsMK);
[ACC,SN,Spec,MCC,auc] = roc( predict_y,Scores,label_2272 )





