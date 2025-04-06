clc;clear;close all

% nfold is number of K-fold cross validation used in objective function
% iter is number of iteration
% ecosize is number of organism in SOS optimziation
% nopt is 3 represents 3 machine learning option provided for SOS to choose
% ubparam and lbparam is upper and lower bound of each machine learning hyperparameters

% Currently this SOS algorithm is design to choose between NN, SVM, or
% ensemble tree, including its hyperparameters that yield lowest error.

% Modified fobjSOS to use other error objective function.

%% INPUT
nfold=5;
iter=5;
ecosize=5;
nopt=3;
ubparam=[5 10 ;1000 1000 ;500 1];
lbparam=[1 4 ;1e-4 1e-4 ;1e-3 1e-4];

%% DATA
data0=xlsread('Concrete_Data.xls','Sheet1');
data0=data0(1:500,:);
perm=randperm(size(data0,1));
data=data0(perm,:);

%% Prepare test and training sets.
ndiv=0.9;
xtrain=data(1:ceil(ndiv*size(data,1)),1:end-1);
ytrain=data(1:ceil(ndiv*size(data,1)),end);

xtest=data(ceil(ndiv*size(data,1))+1:end,1:end-1);
ytest=data(ceil(ndiv*size(data,1))+1:end,end);

%% Optimization
ub=ones(1,nopt)*1;lb=ones(1,nopt)*1e-3;
fitness=zeros(ecosize,1);

eco=rand(ecosize,nopt);
for i=1:ecosize
    fitness(i,:)=fobjSOS(eco(i,:),xtrain,ytrain,nfold,ubparam,lbparam);
end

for h=1:iter
    fprintf('Iteration %d\n',h)
    for i=1:ecosize
        % Update the best Organism
        [bestFitness,idx]=min(fitness); bestOrganism=eco(idx,:);
        save('bestOrganismC.mat','bestOrganism')
        recbestFitness(h)=bestFitness;
        receco{h}=eco;

        %Mutualism Phase
        j=i;
        while i==j
            seed=randperm(ecosize);
            j=seed(1);
        end
        % Determine Mutual Vector & Beneficial Factor
        mutualVector=mean([eco(i,:);eco(j,:)]);
        BF1=round(1+rand); BF2=round(1+rand);
        % Calculate new solution after Mutualism Phase
        ecoNew1=eco(i,:)+rand(1,nopt).*(bestOrganism-BF1.*mutualVector);
        ecoNew2=eco(j,:)+rand(1,nopt).*(bestOrganism-BF2.*mutualVector);
        ecoNew1=bound(ecoNew1,ub,lb);
        ecoNew2=bound(ecoNew2,ub,lb);
        % Evaluate the fitness of the new solution
        fitnessNew1=fobjSOS(ecoNew1,xtrain,ytrain,nfold,ubparam,lbparam);
        fitnessNew2=fobjSOS(ecoNew2,xtrain,ytrain,nfold,ubparam,lbparam);
        % Accept the new solution if the fitness is better
        if fitnessNew1<fitness(i)
            fitness(i)=fitnessNew1;
            eco(i,:)=ecoNew1;
        end
        if fitnessNew2<fitness(j)
            fitness(j)=fitnessNew2;
            eco(j,:)=ecoNew2;
        end
        % End of Mutualism Phase
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Commensialism Phase
        j=i;
        while i==j
            seed=randperm(ecosize);
            j=seed(1);
        end
        % Calculate new solution after Commensalism Phase
        ecoNew1=eco(i,:)+(rand(1,nopt)*2-1).*(bestOrganism-eco(j,:));
        ecoNew1=bound(ecoNew1,ub,lb);
        % Evaluate the fitness of the new solution
        fitnessNew1=fobjSOS(ecoNew1,xtrain,ytrain,nfold,ubparam,lbparam);
        % Accept the new solution if the fitness is better
        if fitnessNew1<fitness(i)
            fitness(i)=fitnessNew1;
            eco(i,:)=ecoNew1;
        end
        % End of Commensalism Phase
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Parasitism Phase
        j=i;
        while i==j
            seed=randperm(ecosize);
            j=seed(1);
        end
        % Determine Parasite Vector & Calculate the fitness
        parasiteVector=eco(i,:);
        seed=randperm(nopt);
        pick=seed(1:ceil(rand*nopt));  % select random dimension
        parasiteVector(:,pick)=rand(1,length(pick)).*(ub(pick)-lb(pick))+lb(pick);
        fitnessParasite=fobjSOS(parasiteVector,xtrain,ytrain,nfold,ubparam,lbparam);

        % Kill organism j and replace it with the parasite
        % if the fitness is lower than the parasite
        if fitnessParasite < fitness(j)
            fitness(j)=fitnessParasite;
            eco(j,:)=parasiteVector;
        end
        % End of Parasitism Phase
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    end
    type=ceil(bestOrganism(1)*size(ubparam,1));
    hyperparam=bestOrganism(2:3).*(ubparam(type,:)-lbparam(type,:))+lbparam(type,:);
    fprintf('Error %d\n',bestFitness)
    if type==1
        fprintf('Best Machine Learning = Artificial Neural Network\n')
        fprintf('Hyperparameter 1 = %d\n',hyperparam(1))
        fprintf('Hyperparameter 2 = %d\n\n',hyperparam(2))
    elseif type==2
        fprintf('Best Machine Learning = Support Vector Machine\n')
        fprintf('Hyperparameter 1 = %d\n',hyperparam(1))
        fprintf('Hyperparameter 2 = %d\n\n',hyperparam(2))
    else
        fprintf('Best Machine Learning = Ensemble Tree\n')
        fprintf('Hyperparameter 1 = %d\n',hyperparam(1))
        fprintf('Hyperparameter 2 = %d\n\n',hyperparam(2))
    end
end