library(readxl)
wkv <- read_excel("Weekly_StkLSTM_Results.xlsx")
mv<- read_excel("Monthly_StkLSTM_Results.xlsx")
tv<- read_excel("Trimester_StkLSTM_Results.xlsx")
test<-rbind(wkv,mv, tv)

Timeline<-1:108
Timeline[1:36]<-"Weekly"
Timeline[37:72]<-"Monthly"
Timeline[73:108]<-"Quarterly"

test$Timeline<-cbind(Timeline)

library(dplyr)
library(ggplot2)
test$`Neurons Layer 1` = as.factor(test$`Neurons Layer 1`)
test$`Neurons Layer 2` = as.factor(test$`Neurons Layer 2`)
test$Batches=as.factor(test$Batches)
test$Epochs=as.factor(test$Epochs)
test$Timeline=as.factor(test$Timeline)


library(stringr)

# weekly plots by number of neurons on the first
test%>%filter(`Neurons Layer 1`==32, Timeline=="Weekly")%>%
  ggplot(aes(x=Epochs, y=MAPE,fill=`Neurons Layer 2`))+ 
  geom_bar(stat = "identity", position="dodge") + 
  facet_grid(Timeline~Batches) + theme_light()+
  geom_text(aes(label = MAPE), size=4.5,
            position = position_dodge(width = 0.9), 
            vjust = -.25)+
  theme(text = element_text(size=20),
        axis.text.x = element_text(color = "grey20",size = 20,hjust = .5, 
                                   vjust = .5, face = "plain"))+
  labs(x="Epochs", y= "MAPE (%)", 
       fill=str_wrap("Neurons in Second Layer",20))+
  scale_fill_brewer()

test%>%filter(`Neurons Layer 1`==64, Timeline=="Weekly")%>%
  ggplot(aes(x=Epochs, y=MAPE,fill=`Neurons Layer 2`))+ 
  geom_bar(stat = "identity", position="dodge") + 
  facet_grid(Timeline~Batches) + theme_light()+
  geom_text(aes(label = MAPE), size=4.5,
            position = position_dodge(width = 0.9), 
            vjust = -.25)+
  theme(text = element_text(size=20),
        axis.text.x = element_text(color = "grey20",size = 20,hjust = .5, 
                                   vjust = .5, face = "plain"))+
  labs(x="Epochs", y= "MAPE (%)", 
       fill=str_wrap("Neurons in Second Layer",20))+
  scale_fill_brewer()

# Monthly plots by neurons 1
test%>%filter(`Neurons Layer 1`==32, Timeline=="Monthly")%>%
  ggplot(aes(x=Epochs, y=MAPE,fill=`Neurons Layer 2`))+ 
  geom_bar(stat = "identity", position="dodge") + 
  facet_grid(Timeline~Batches) + theme_light()+
  geom_text(aes(label = MAPE), size=4.5,
            position = position_dodge(width = 0.9), 
            vjust = -.25)+
  theme(text = element_text(size=20),
        axis.text.x = element_text(color = "grey20",size = 20,hjust = .5, 
                                   vjust = .5, face = "plain"))+
  labs(x="Epochs", y= "MAPE (%)", 
       fill=str_wrap("Neurons in Second Layer",20))+
  scale_fill_brewer()

test%>%filter(`Neurons Layer 1`==64, Timeline=="Monthly")%>%
  ggplot(aes(x=Epochs, y=MAPE,fill=`Neurons Layer 2`))+ 
  geom_bar(stat = "identity", position="dodge") + 
  facet_grid(Timeline~Batches) + theme_light()+
  geom_text(aes(label = MAPE), size=4.5,
            position = position_dodge(width = 0.9), 
            vjust = -.25)+
  theme(text = element_text(size=20),
        axis.text.x = element_text(color = "grey20",size = 20,hjust = .5, 
                                   vjust = .5, face = "plain"))+
  labs(x="Epochs", y= "MAPE (%)", 
       fill=str_wrap("Neurons in Second Layer",20))+
  scale_fill_brewer()

#quarterly plots
test%>%filter(`Neurons Layer 1`==32, Timeline=="Quarterly")%>%
  ggplot(aes(x=Epochs, y=MAPE,fill=`Neurons Layer 2`))+ 
  geom_bar(stat = "identity", position="dodge") + 
  facet_grid(Timeline~Batches) + theme_light()+
  geom_text(aes(label = MAPE), size=4.5,
            position = position_dodge(width = 0.9), 
            vjust = -.25)+
  theme(text = element_text(size=20),
        axis.text.x = element_text(color = "grey20",size = 20,hjust = .5, 
                                   vjust = .5, face = "plain"))+
  labs(x="Epochs", y= "MAPE (%)", 
       fill=str_wrap("Neurons in Second Layer",20))+
  scale_fill_brewer()

test%>%filter(`Neurons Layer 1`==64, Timeline=="Quarterly")%>%
  ggplot(aes(x=Epochs, y=MAPE,fill=`Neurons Layer 2`))+ 
  geom_bar(stat = "identity", position="dodge") + 
  facet_grid(Timeline~Batches) + theme_light()+
  geom_text(aes(label = MAPE), size=4.5,
            position = position_dodge(width = 0.9), 
            vjust = -.25)+
  theme(text = element_text(size=20),
        axis.text.x = element_text(color = "grey20",size = 20,hjust = .5, 
                                   vjust = .5, face = "plain"))+
  labs(x="Epochs", y= "MAPE (%)", 
       fill=str_wrap("Neurons in Second Layer",20))+
  scale_fill_brewer()


