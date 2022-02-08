library(dplyr)   # For data manipulation
library(ggplot2) # For data visualization
library(reshape)
library(ggforce)
library(ggpubr)
library(ggrepel)
library(forcats)
library(ggridges)
library(latex2exp)

theme_set(theme_pubclean() +
          theme(panel.background = element_rect(fill = "white", colour = "black")))

data <- read.csv('./diff.csv')
data <- data[complete.cases(data), ]
data['Category'] <- ifelse(data$True<0.001, "Metal", "Non-metal")
data$Fidelity <- factor(data$Fidelity,
                           levels=c("PBE", 'GLLB-SC', 'SCAN', 'HSE', 'Exp'))
data$Category <- factor(data$Category, levels=c("Metal", "Non-metal"))
data$Fill <- paste(data$Fidelity, data$Category)

new_levels <- as.vector(t(outer(levels(data$Fidelity), levels(data$Category), paste)))
data$Fill <- factor(data$Fill, levels=new_levels)

r1 <- "#ff0000"
b1 <- "#0000ff"
r2 <- '#ff8080'
b2 <- '#8080ff'

p <- ggplot(data, aes(y=Fidelity)) +
  geom_density_ridges(aes(x=Diff, fill=Fill),
                      alpha=0.8, color='white', from = -1.5, to = 1.5, scale=1)  +
  labs(x = TeX("$\\textit{\\Delta}$ (eV)"), y='') +
  scale_y_discrete(expand=c(0, 0)) +
  scale_fill_cyclical(breaks = c("PBE Metal", "PBE Non-metal"),
                      labels = c(`PBE Metal` = "Metal", `PBE Non-metal` = "Non-metal"),
                      values=c(r1, b1, b2,
                               r1, b1, r2, b2,
                               r1, b1, r2, b2),
                      name='', guide='legend') +
  facet_grid(.~Category) +
  coord_cartesian(clip = 'off') +
  theme_ridges(grid=FALSE) +
  theme(axis.title.x = element_text(hjust=0.5),
        legend.position = c(0.35, 0.96),
        axis.line.x = element_line(color='black', size=0.3, linetype='solid'),
        axis.text.y = element_blank(),
        strip.text = element_blank(),
        panel.spacing = unit(2, "lines"),)


print(p)
ggsave('diff.pdf', plot=p, height = 4, width = 4.2)
