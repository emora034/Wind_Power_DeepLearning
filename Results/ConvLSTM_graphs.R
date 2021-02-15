library(readxl)
wkv <- read_excel("Weekly_CNNLSTM_Results.xlsx")
mv<- read_excel("Monthly_CNNLSTM_Results.xlsx")
tv<- read_excel("Trimester_CNNLSTM_Results.xlsx")
#wkv<-rbind(wkv,mv,tv)

Timeline<-1:27
Timeline[1:27]<-"Weekly"
#Timeline[28:54]<-"Monthly"
wkv$Timeline<-cbind(Timeline)

library(dplyr)
library(ggplot2)
wkv$Neurons = as.factor(wkv$Neurons)
wkv$Batches=as.factor(wkv$Batches)
wkv$Epochs=as.factor(wkv$Epochs)
wkv$Timeline=as.factor(wkv$Timeline)


wkv%>%filter(Timeline=="Weekly")%>% ggplot(aes(x=Epochs, y=MAPE,fill=Neurons))+ 
  geom_bar(stat = "identity", position="dodge") + 
  facet_grid(Timeline~Batches) + theme_light()+
  geom_text(aes(label = MAPE), size=4.5,
            position = position_dodge(width = 0.9), 
            vjust = -.25)+
  theme(text = element_text(size=20),
        axis.text.x = element_text(color = "grey20",size = 20,hjust = .5, 
                                   vjust = .5, face = "plain"))+
  labs(x="Epochs", y= "MAPE (%)")+
  scale_fill_brewer()


# TRIMESTER PLOTS
Timeline<-1:9
Timeline[1:9]<-"Quarterly"

tv$Timeline<-cbind(Timeline)

tv$Neurons = as.factor(tv$Neurons)
tv$Batches=as.factor(tv$Batches)
tv$Epochs=as.factor(tv$Epochs)
tv$Timeline=as.factor(tv$Timeline)

tv%>%filter(Timeline=="Quarterly")%>% ggplot(aes(x=Epochs, y=MAPE,fill=Neurons))+ 
  geom_bar(stat = "identity", position="dodge") + 
  facet_grid(Timeline~Batches) + theme_light()+
  geom_text(aes(label = MAPE), size=4.5,
            position = position_dodge(width = 0.9), 
            vjust = -.25)+
  theme(text = element_text(size=20),
        axis.text.x = element_text(color = "grey20",size = 20,hjust = .5, 
                                   vjust = .5, face = "plain"))+
  labs(x="Epochs", y= "MAPE (%)")+
  scale_fill_brewer()

# monthly PLOTS
Timeline<-1:9
Timeline[1:9]<-"Monthly"

mv$Timeline<-cbind(Timeline)

mv$Neurons = as.factor(mv$Neurons)
mv$Batches=as.factor(mv$Batches)
mv$Epochs=as.factor(mv$Epochs)
mv$Timeline=as.factor(mv$Timeline)

mv%>%filter(Timeline=="Monthly")%>% ggplot(aes(x=Epochs, y=MAPE,fill=Neurons))+ 
  geom_bar(stat = "identity", position="dodge") + 
  facet_grid(Timeline~Batches) + theme_light()+
  geom_text(aes(label = MAPE), size=4.5,
            position = position_dodge(width = 0.9), 
            vjust = -.25)+ 
  theme(text = element_text(size=20),
        axis.text.x = element_text(color = "grey20",size = 20,hjust = .5, 
                                   vjust = .5, face = "plain"))+
  labs(x="Epochs", y= "MAPE (%)")+
  scale_fill_brewer()

