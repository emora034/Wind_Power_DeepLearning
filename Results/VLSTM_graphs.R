library(readxl)
wkv <- read_excel("Weekly_VanSLTM.xlsx")
mv<- read_excel("Monthly_VanLSTM_Results.xlsx")
tv<- read_excel("Trimester_VanLSTM_Results.xlsx")
test<-rbind(wkv,mv,tv)

Timeline<-1:81
Timeline[1:27]<-"Weekly"
Timeline[28:54]<-"Monthly"
Timeline[55:81]<-"Quarterly"

test$Timeline<-cbind(Timeline)

library(dplyr)
library(ggplot2)
test$Neurons = as.factor(test$Neurons)
test$Batches=as.factor(test$Batches)
test$Epochs=as.factor(test$Epochs)
test$Timeline=as.factor(test$Timeline)


test%>%filter(Timeline=="Weekly")%>% ggplot(aes(x=Epochs, y=MAPE,fill=Neurons))+ 
  geom_bar(stat = "identity", position="dodge") + 
  facet_grid(Timeline~Batches) + theme_light()+
  geom_text(aes(label = MAPE), size=4.5,
            position = position_dodge(width = 0.9), 
            vjust = -.25)+
  labs(x="Epochs", y= "MAPE (%)")+
  theme(text = element_text(size=20),axis.text.x = element_text(color = "grey20", size = 20, 
  hjust = .5, vjust = .5, face = "plain"))+
  labs(x="Epochs", y= "MAPE (%)")+
  scale_fill_brewer()


test%>%filter(Timeline=="Monthly")%>% ggplot(aes(x=Epochs, y=MAPE,fill=Neurons))+ 
  geom_bar(stat = "identity", position="dodge") + 
  facet_grid(Timeline~Batches) + theme_light()+
  geom_text(aes(label = MAPE), size=4.5,
            position = position_dodge(width = 0.9), 
            vjust = -.25)+
  theme(text = element_text(size=20),axis.text.x = element_text(color = "grey20", 
        size = 20,hjust = .5, vjust = .5, face = "plain"))+
  labs(x="Epochs", y= "MAPE (%)")+
  scale_fill_brewer()

test%>%filter(Timeline=="Quarterly")%>% ggplot(aes(x=Epochs, y=MAPE,fill=Neurons))+ 
  geom_bar(stat = "identity", position="dodge") + 
  facet_grid(Timeline~Batches) + theme_light()+
  geom_text(aes(label = MAPE), size=4.5,
            position = position_dodge(width = 0.9), 
            vjust = -.25)+
  theme(text = element_text(size=20),axis.text.x = element_text(color = "grey20", 
            size = 20,hjust = .5, vjust = .5, face = "plain"))+
  labs(x="Epochs", y= "MAPE (%)")+
  scale_fill_brewer()



