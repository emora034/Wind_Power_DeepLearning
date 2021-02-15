library(readxl)
test<-read_excel("Weekly_AutoLSTM_Results.xlsx")

library(dplyr)
library(ggplot2)
test$Neurons= as.factor(test$Neurons)
test$Batches=as.factor(test$Batches)
test$Epochs=as.factor(test$Epochs)


steps<-1:27
steps[1:9]<-'1 Step'
steps[10:18]<-'2 Steps' 
steps[19:27]<-'3 Steps'
steps<-as.factor(steps)
RMSE<-c(test$`TestRMSE 1`, test$`TestRMSE 2`, test$`TestRMSE 3`)
MAPE<-c(test$`MAPE 1`, test$`MAPE 2`, test$`MAPE 3`)
Neurons<-test$Neurons
Epochs<-test$Epochs
Batches<-test$Batches
df<-data.frame(steps,RMSE, MAPE,
               Neurons, Epochs, Batches)

ggplot(df, aes(x=Epochs, y=MAPE, fill=Neurons))+
  theme(legend.title=element_blank(),
        legend.position = 'none')+
  geom_bar( stat = "identity", position="dodge")+
  facet_grid(steps~Batches) + theme_light()+
  geom_text(aes(label = MAPE), size=4.5,
            position = position_dodge(width = 0.9), 
            vjust = -.25)+ theme_light()+
  theme(text = element_text(size=20),axis.text.x = element_text(color = "grey20", 
        size = 20,hjust = .5, vjust = .5, face = "plain"))+
  labs(x='Epochs', y= "MAPE (%)")+
  scale_fill_brewer(palette='Blues')+ ylim(0,15)

ggplot(df, aes(x=Epochs, y=RMSE, fill=Neurons))+
  theme(legend.title=element_blank(),
        legend.position = 'none')+
  geom_bar( stat = "identity", position="dodge")+
  facet_grid(steps~Batches) + theme_light()+
  geom_text(aes(label = RMSE), size=4.5,
            position = position_dodge(width = 0.9), 
            vjust = -.25)+ theme_light()+
  theme(text = element_text(size=20),axis.text.x = element_text(color = "grey20", 
       size = 20,hjust = .5, vjust = .5, face = "plain"))+
  labs(x='Epochs', y= "RMSE (MWh)")+
  scale_fill_brewer(palette='OrRd')+ ylim(0,800)
