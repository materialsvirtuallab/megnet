library(dplyr)   # For data manipulation
library(ggplot2) # For data visualization
library(reshape)
library(ggforce)
library(ggpubr)
library(ggrepel)

theme_set(theme_pubclean() +
          theme())

data <- read.csv('./metal_vs_nonmetal.csv')
data <- data[, c("Fidelity", "Metal" , "Non.metal", "Metal.clip")]

data <- melt(data)
data <- data[complete.cases(data), ]
colnames(data) <- c('Fidelity', 'Category', "Error")

avg <- data %>%
  group_by(Fidelity, Category) %>%
  summarise(mean = mean(Error),
            q = quantile(Error, 0.75),
            ypos = mean(Error) +0.04)

avg[avg$Fidelity=='PBE', 'ypos'] = avg[avg$Fidelity=='PBE', 'ypos'] - 0.01
avg[(avg$Fidelity=='HSE') & (avg$Category=='Non.metal'), 'ypos'] =
  avg[(avg$Fidelity=='HSE') & (avg$Category=='Non.metal'), 'ypos'] + 0.015

avg[(avg$Fidelity=='SCAN') & (avg$Category=='Non.metal'), 'ypos'] =
  avg[(avg$Fidelity=='SCAN') & (avg$Category=='Non.metal'), 'ypos'] -0.01


avg$Fidelity = factor(avg$Fidelity, levels=c("PBE", "GLLB-SC", "HSE", "SCAN", "Exp"))
avg$Category = factor(avg$Category, levels=c("Metal", "Metal.clip", "Non.metal"))

p <- ggplot(avg, aes(x = Category, y = mean)) +
  geom_linerange(aes(x = Category, ymin = 0, ymax = mean,
                     group=factor(Fidelity, levels=c("PBE", "GLLB-SC", "HSE", "SCAN", "Exp"))),
                 color='lightgray', size=1.5, position = position_dodge(0.8)) +
  geom_point(aes(color=factor(Fidelity, levels=c("PBE", "GLLB-SC", "HSE", "SCAN", "Exp"))), position=position_dodge(0.8), size=3) +
  scale_x_discrete(labels=c("Metal" = "Metal",
                            "Metal.clip" = "Metal-clip",
                           "Non.metal" = "Non-metal"
                           )) +
  geom_text(aes(x = Category, y = ypos, label=round(mean, 2), color=factor(Fidelity, levels=c("PBE", "GLLB-SC", "HSE", "SCAN", "Exp"))),
            position=position_dodge(0.8)) +
  coord_cartesian(ylim = c(0, 0.7))+
  xlab("") + ylab("MAE (eV)") +
  theme(legend.position = c(0.2, 0.8),
        legend.title = element_blank(),
        legend.text = element_text(size=10),
        legend.background = element_rect(fill=alpha('white', 0)),
        text = element_text(size=15, color='black'),
        axis.text = element_text(color='#000000'),
        panel.grid = element_blank())

print(p)
ggsave('exp_is_metal.pdf', plot=p, height = 4, width = 4.2)
