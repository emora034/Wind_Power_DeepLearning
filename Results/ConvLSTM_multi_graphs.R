library(readxl)
test<- read_xlsx("CNN_MultiLSTM_Results.xlsx")


library(dplyr)
library(ggplot2)
test$Neurons = as.factor(test$Neurons)
test$Batches=as.factor(test$Batches)
test$Epochs=as.factor(test$Epochs)



steps<-c('1 Step', '2 Steps', '3 Steps')
RMSE<-c(test$`TestRMSE 1`, test$`TestRMSE 2`, test$`TestRMSE 3`)
MAPE<-c(test$`MAPE 1`, test$`MAPE 2`, test$`MAPE 3`)
df<-data.frame(as.factor(steps),RMSE, MAPE )

p1<-ggplot(df, aes(x=steps, y=MAPE, fill=steps))+
  theme(legend.title=element_blank(),
        legend.position = 'none')+
  geom_bar( stat = "identity", position="dodge")+
  geom_text(aes(label = MAPE),  size=5,
            position = position_dodge(width = 0.9), 
            vjust = -.25)+ theme_light()+
  theme(text = element_text(size=20),axis.text.x = element_text(color = "grey20", 
        size = 20,hjust = .5, vjust = .5, face = "plain"))+
  labs(x='Predicted Time Steps', y= "MAPE (%)")+
  scale_fill_brewer(palette='Blues')+
  theme(legend.title=element_blank(), legend.position = 'none')

p2<-ggplot(df, aes(x=steps, y=RMSE, fill=steps))+
  geom_bar(stat = "identity", position="dodge")+
  geom_text(aes(label = RMSE), size=5,
            position = position_dodge(width = 0.9), 
            vjust = -.25)+ theme_light()+
  theme(text = element_text(size=20),axis.text.x = element_text(color = "grey20", 
        size = 20,hjust = .5, vjust = .5, face = "plain"))+
  labs(x='Predicted Time Steps',y= "RMSE (MWh)")+
  scale_fill_brewer(palette = 'OrRd')+
  theme(legend.title=element_blank(), legend.position = 'none')

library(patchwork)
p1+p2
