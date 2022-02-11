library(dplyr)   # For data manipulation
library(ggplot2) # For data visualization
library(reshape2)

theme_set(theme(panel.background = element_rect(fill = "white", colour = "black")))

data <- read.csv('./all_errors.csv', stringsAsFactors=FALSE)
data$Fidelity = factor(data$Fidelity, levels=c("PBE", "GLLB-SC", "SCAN", "HSE", "Exp"))

## include some dummy data to empty entries
fake_lines <- data.frame(Model=c("2-fi", "4-fi", "1-fi-stacked"),
                         Fidelity=c("PBE", "SCAN", "PBE"),
                         Error=c(1000, 1000, 1000),
                         split=c(0, 0, 0))

data = rbind(data, fake_lines)

p <- ggplot(data, aes(x=Model, y=Error, fill=Model)) +
  geom_boxplot(width = 5, color="black") +
#  geom_jitter( width=0.1,alpha=0.9) +  # add dot plots
  xlab('') + ylab('MAE (eV)') +
  scale_x_discrete(labels=c("GLLBSC" = "GLLB-SC"), drop=FALSE,
  ) +
  coord_cartesian(ylim = c(0.,1))+
  facet_grid(. ~ Fidelity, scales = "free", space = "free") +
  theme(
    axis.text=element_text(size=16, color='#000000'),
    axis.title=element_text(size=16),
    legend.text=element_text(size=16),
    legend.title=element_blank(),
    legend.background = element_rect(fill=alpha('white', 0), color=alpha('white', 0)),
    legend.position = c(0.1, 0.8),
    panel.spacing = unit(0, "lines"),
    strip.background = element_rect(fill='white', color='white'),
    strip.text=element_blank(),

  ) +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) +
  geom_text(aes(label = Fidelity), x = Inf, y = Inf, hjust = 1.1, vjust = 1.5,
             size=5, check_overlap=TRUE)


#ggsave('figure2a_box_with_dots.pdf', plot=p, height = 4.5, width = 9)

ggsave('figure2a_box.pdf', plot=p, height = 4.5, width = 9)
print(p)
