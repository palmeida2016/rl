data = table2array(readtable('test.csv'));
movingMean = movmean(data(:,2), [1 25000]);
coeff = polyfit(data(:,1),movingMean,1);
y = polyval(coeff, data(:,1));
plot(data(:,1),movingMean,data(:,1),y);
legend()