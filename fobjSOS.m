function G=fobjSOS(x,xtrain,ytrain,nfold,ubparam,lbparam)
type=ceil(x(1)*size(ubparam,1));
x(2:3)=x(2:3).*(ubparam(type,:)-lbparam(type,:))+lbparam(type,:);

for h=1:nfold
    dummy1=1:size(ytrain,1)/nfold:size(ytrain,1)+1;
    dummy2=zeros(size(ytrain,1),1);
    dummy2(dummy1(h):1:dummy1(h+1)-1)=1;
    dummy3=abs(dummy2-1);

    dummy2=logical(dummy2);
    dummy3=logical(dummy3);

    XTrain=xtrain(dummy3,:);
    YTrain=ytrain(dummy3,:);
    XTest=xtrain(dummy2,:);
    YTest=ytrain(dummy2,:);

    if type==1
        x=ceil(x);
        Mdl1=fitrnet(XTrain,YTrain,"Standardize",true,"LayerSizes",ones(1,x(2)).*x(3),'Activations','relu',"Verbose",0);
        Ypred=predict(Mdl1,XTest);
    elseif type==2
        Mdl2=fitrsvm(XTrain,YTrain,'Standardize',true,'BoxConstraint',x(2),'KernelFunction','rbf','KernelScale',x(3),'Epsilon',0,'Verbose',0);
        Ypred=predict(Mdl2,XTest);
    else
        Mdl3=fitrensemble(XTrain,YTrain,'NumLearningCycles',x(2),'LearnRate',x(3));
        Ypred=predict(Mdl3,XTest);
    end
    MAPE=rms(Ypred-YTest);
    dummy=corrcoef(Ypred,YTest);
    R=dummy(1,2);
    g(h)=MAPE+(1-R);
end
G=mean(g);