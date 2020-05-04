library(glmnet)

Xstan = read.csv("c:/htoc/biof309/final-project/data/Xstan.csv")

y = Xstan$y
Xstan = Xstan[,-1]
Xstan = Xstan[,-which(names(Xstan)=="y")]

glmmod <- glmnet(as.matrix(Xstan), y=as.factor(y), alpha=1, family="binomial",standardize=FALSE)

cv.glmmod <- cv.glmnet(as.matrix(Xstan), y=as.factor(y), alpha=1, family="binomial",standardize=FALSE)

plot(cv.glmmod)

(best.lambda <- cv.glmmod$lambda.min)


linpreds =  predict.glmnet(glmmod, s = best.lambda,newx=as.matrix(Xstan))
pees = exp(linpreds)/(1+exp(linpreds))

table(ifelse(pees>.5,1,0))

summary(glm(y ~ Xstan,family="binomial"))

age + distancefromhome + monthlyincome          
numcompaniesworked + percentsalaryhike + totalworkingyears + yearsatcompany + yearssincelastpromotion + yearswithcurrmanager +    
travelalot              randddepartment         sciencemedicaleduc     
male                    researchjob             evermarried            
highworkenvironment     highjobsatisfaction     highworklifebalance    