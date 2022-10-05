
pathDir = pwd;
pathDataset = strcat(pwd,'/dataset');
pathResults = strcat(pwd,'/kNNResults');
pathResultsMAT = strcat(pwd,'/kNNResults/');

simName = 'kNNRobustness';
logName = strcat(pathResults,'/kNNRobustness.txt');

fileID = fopen(logName,'a');
fprintf(fileID, 'kNNRobustness\n');
fclose(fileID);

noK = 4;

kList= [100 200 300 400]';


shuffledIndices = [62 40 95 18 97 84 64 42 10  0 31 76 47 26 44  4 22 12 88 73 49 70 68 15 ...
    39 33  9 81 11 65 94 30 28 89  5 45 69 35 16 72 34  7 55 27 19 80 25 53 ...
    13 24  3 17 38  8 77  6 79 36 91 56 98 54 43 50 66 46 67 61 96 78 41 58 ...
    48 85 57 75 32 93 59 63 83 37 29  1 52 21  2 23 87 90 74 86 82 20 60 71 ...
    14 92 51];
shuffledIndices = shuffledIndices + 1;


mapTra = shuffledIndices(1:84);
noMapTra = 84;
mapTest = shuffledIndices(85:99);
noMapTest = 15;

noRx = 200;
noTrials = 50;

load(strcat(pathDataset,'/TxRx'),'rxx','rxy','antX','antY','antList')

p = gcp; %get parallel pool for parallel computing in CPU
Workers=2*p.NumWorkers; %number of workers is twice the number of CPU cores

difTra = zeros(noTrials,noMapTra,noRx,noK);
difTest = zeros(noTrials,noMapTest,noRx);

tic
for tt=1:noTrials
    
    for k=1:noK
        
        kk = kList(k);
        for mm=1:noMapTra
            
            shuffledM = mapTra(mm);
            parfor (rx=1:noRx, Workers)
                %             for rx=1:noRx
                
                difTra(tt,mm,rx,k) = kNNRobustnessFunCluster(kk,shuffledM,squeeze(antList(tt,shuffledM,:)),squeeze(rxx(shuffledM,rx)),squeeze(rxy(shuffledM,rx)));
            end
        end
    end
    save(strcat(pathResultsMAT,simName,'Tra'),'difTra')
    disp('tx')
    disp('One trial training')
    toc
end

%Get MAE for each k
MAETraK = zeros(noK,1);


for k=1:noK
    kVal = kList(k);
    MAEkAll = difTra(:,:,:,k);
    MAEkAllFlat = MAEkAll(:);
    difAllNaNs = find(isnan(MAEkAllFlat));
    difAllNoNaNs = MAEkAllFlat;
    difAllNoNaNs(difAllNaNs) = [];
    MAETraK(k) = sum(difAllNoNaNs)/numel(difAllNoNaNs);
    
    fileID = fopen(logName,'a');
    a=sprintf('%.2f',MAETraK(k));
    fprintf(fileID, strcat('k = :',int2str(kVal),', ','MAE: = ',a,'\n'));
    fclose(fileID);
    
end

[minMAE bestKind] = min(MAETraK);
bestK = kList(bestKind);

fileID = fopen(logName,'a');
fprintf(fileID, strcat('Best k = :',int2str(bestK),'\n'));
fclose(fileID);

%Test with found k

for tt=1:noTrials
    for mm=1:noMapTest
        
        shuffledM = mapTest(mm);
        parfor (rx=1:noRx, Workers)
            %         for rx=1:noRx
            difTest(tt,mm,rx) = DPMtoIRT2NoisyFunCluster(bestK,shuffledM,squeeze(antList(tt,shuffledM,:)),squeeze(rxx(shuffledM,rx)),squeeze(rxy(shuffledM,rx)));
        end
        
    end
    
    save(strcat(pathResultsMAT,simName,'Test'),'difTest')
    
end

MAETestAll =difTest;
MAETestFlat = MAETestAll(:);
difAllNaNs = find(isnan(MAETestFlat));
difAllNoNaNs = MAETestFlat;
difAllNoNaNs(difAllNaNs) = [];
%mae
MAETest= sum(difAllNoNaNs)/numel(difAllNoNaNs);

a=sprintf('%.2f',MAETest);

%rmse
VarTest = sum(difAllNoNaNs.^2)/numel(difAllNoNaNs);
RMSETest = sqrt(VarTest);

r=sprintf('%.2f',RMSETest);
fileID = fopen(logName,'a');
fprintf(fileID, strcat('Test k = :',int2str(bestK),', ','Test MAE: = ',a,', ','RMSE: = ',r ,'\n'));
fclose(fileID);


fileID = fopen(logName,'a');
a=sprintf('%.2f',sqrt(RMSETest));
fprintf(fileID, strcat('RMSE: = ',a,'\n'));
fclose(fileID);


