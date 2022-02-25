function [ACC,SN,Spec,PE,NPV,F_score,MCC,auc] = roc( predict_label,Scores,test_data_label )
%ROC Summary of this function goes here
%   Detailed explanation goes here
l=length(predict_label);
TruePositive = 0;
TrueNegative = 0;
FalsePositive = 0;
FalseNegative = 0;
for k=1:l
    if test_data_label(k)==1 & predict_label(k)==1  %真阳性
        TruePositive = TruePositive +1;
    end
    if test_data_label(k)==-1 & predict_label(k)==-1 %真阴性
        TrueNegative = TrueNegative +1;
    end 
    if test_data_label(k)==-1 & predict_label(k)==1  %假阳性
        FalsePositive = FalsePositive +1;
    end

    if test_data_label(k)==1 & predict_label(k)==-1  %假阴性
        FalseNegative = FalseNegative +1;
    end
end
%TruePositive
%TrueNegative
%FalsePositive
%FalseNegative
ACC = (TruePositive+TrueNegative)./(TruePositive+TrueNegative+FalsePositive+FalseNegative);
SN = TruePositive./(TruePositive+FalseNegative);

Spec = TrueNegative./(TrueNegative+FalsePositive);%

PE=TruePositive./(TruePositive+FalsePositive);

NPV = TrueNegative./(TrueNegative+FalseNegative);

F_score = 2*(SN*PE)./(SN+PE);

MCC= (TruePositive*TrueNegative-FalsePositive*FalseNegative)./sqrt(  (TruePositive+FalseNegative)...
    *(TrueNegative+FalsePositive)*(TruePositive+FalsePositive)*(TrueNegative+FalseNegative));

%[X_value,Y_value]= roc_curve( predict_label,test_data_label);
%auc = -trapz(X_value,Y_value);
auc = roc_curve( Scores,test_data_label);
end


function auc = roc_curve(deci,label_y) %%deci=wx+b, label_y, true label
    [val,ind] = sort(deci,'descend');
    roc_y = label_y(ind);
    stack_x = cumsum(roc_y == -1)/sum(roc_y == -1);
    stack_y = cumsum(roc_y == 1)/sum(roc_y == 1);
    auc = sum((stack_x(2:length(roc_y),1)-stack_x(1:length(roc_y)-1,1)).*stack_y(2:length(roc_y),1))
 
%     plot(stack_x,stack_y);
%     xlabel('False Positive Rate');
%     ylabel('True Positive Rate');
%     title(['ROC curve of (AUC = ' num2str(auc) ' )']);
end