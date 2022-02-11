library(dplyr)   # For data manipulation
library(ggplot2) # For data visualization
library(reshape2)
library(RColorBrewer)

theme_set(theme(panel.background = element_rect(fill = "white", colour = "black")))

data <- read.csv('./all_errors.csv', stringsAsFactors=FALSE)
data$Fidelity = factor(data$Fidelity, levels=c("PBE", "GLLB-SC", "SCAN", "HSE", "Exp"))
data[data['Model'] == '1-fi-stacked', "Model"] = '1-fi-s'
# data$Model = factor(data$Model, levels=c("1-fi", "1-fi-s", "2-fi", "4-fi", "5-fi"))
data$Model = factor(data$Model, levels=c("5-fi", "4-fi", "2-fi", "1-fi-s", "1-fi"))
p <- ggplot(data, aes(x=Model, y=Error, fill=Model)) +
  geom_boxplot(width = 5, color="black") +
  scale_fill_brewer(palette="YlGnBu", guide=guide_legend(reverse=TRUE)) +
  geom_jitter( width=0.1,alpha=0.9) +  # add dot plots
  xlab('') + ylab('MAE (eV)') +
  scale_x_discrete(labels=c("GLLBSC" = "GLLB-SC"), drop=FALSE,
  ) +
  ylim(c(0, 1.1)) +
  facet_grid(vars(Fidelity)) +
  theme(
    axis.text=element_text(size=16, color='#000000'),
    axis.title=element_text(size=16),
    axis.ticks.y=element_blank(),
    axis.text.y=element_blank(),
    axis.title.y=element_blank(),
    legend.box = "horizontal",
    legend.position="top",
    legend.text=element_text(size=16),
    legend.title=element_blank(),
    legend.background = element_rect(fill=alpha('white', 0), color=alpha('white', 0)),
    #legend.position = c(0.1, 0.8),
    panel.spacing = unit(0, "lines"),
    strip.background = element_rect(fill='white', color='white'),
    strip.text=element_blank()
  ) +
  coord_flip() +

#  theme(axis.title.x=element_blank(),
#        axis.text.x=element_blank(),
#        axis.ticks.x=element_blank()) +
  geom_text(aes(label = Fidelity), x = Inf, y = 0, hjust = 0, vjust = 2,
            size=5, check_overlap=TRUE)


# ggsave('two_fidelity_with_dots.pdf', plot=p, height = 4.5, width = 9)

ggsave('mf_dots.pdf', plot=p, height = 4.5, width = 8)
print(p)
