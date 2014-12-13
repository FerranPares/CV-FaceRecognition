function Rates=validation(features, labels, subjects, F)
% VALIDATION: Validation of the classification method using F fold cross validation
% Outputs: 
% Rates, structure storing the validation rates.
%k=5;
% Storage 
Storage = [];
% Loop for k neightboors
for k=[1 3 5 7 9]
    % Fold validation strategy
    [error,FP,FN,TP,TN] = fold_validation(features, labels, subjects, F, k);

    % Compute and store the mean rates of validation in
    error=mean(error);
    FP=mean(FP);
    FN=mean(FN);
    TP=mean(TP);
    TN=mean(TN);
    
    Storage = [Storage;error,FP,FN,TP,TN];
    
end


%Plot error for each k used
p=[1 3 5 7 9];
ploterrors = Storage(:,1)';
figure;
plot(p, ploterrors, 'g');
xlabel('k value used');
ylabel('Error rate percentage');

Storagemean = mean(Storage);
error = Storagemean(1);
FP = Storagemean(2);
FN = Storagemean(3);
TP = Storagemean(4);
TN = Storagemean(5);

Rates.Sens=TP/(TP+FN)*100;
Rates.Spec=TN/(TN+FP)*100;
Rates.Prec=TP/(TP+FP)*100;
Rates.FAR=FP/(TP+FN)*100;
Rates.Recall=TP/(TP+FN)*100;
Rates.Acc=(TP+TN)/(TP+TN+FP+FN)*100;
Rates.Error=error;
Rates.ConfusionMatrix=[TP FN; FP TN];



