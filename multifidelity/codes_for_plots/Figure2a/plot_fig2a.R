library(dplyr)   # For data manipulation
library(ggplot2) # For data visualization
library(reshape2)

theme_set(theme_pubclean() + 
            theme(panel.background = element_rect(fill = "white", colour = "black")))

data <- read.csv('./data.csv', stringsAsFactors=FALSE)
error <- read.csv('./error.csv', stringsAsFactors=FALSE)

data <- melt(data, id='name', value.name='mean', variable.name = 'Fidelity')
error <- melt(error, id='name', value.name='std', variable.name = 'Fidelity')

df <- merge(data, error)
df[, 'mean'] <- as.numeric(df[, 'mean'])
df[, 'std'] <- as.numeric(df[, 'std'])
p <- ggplot(df, aes(x=Fidelity, y=mean)) +
  geom_bar(stat = "identity", aes(fill=name), 
           position = position_dodge(width = 1), alpha=0.4) + 
  geom_errorbar(aes(ymin = mean - std, ymax = mean + std, color=name), 
                width = 0.5,
                position = position_dodge(width = 1)) + 
  xlab('') + ylab('MAE (eV)') + 
  scale_x_discrete(labels=c("GLLBSC" = "GLLB-SC")
  ) + 
  theme(
    axis.text=element_text(size=16, color='#000000'),
    axis.title=element_text(size=16),
    legend.text=element_text(size=16), 
    legend.title=element_blank(),
    legend.position=c(0.12, 0.85),
  )

ggsave('figure3a_bar.pdf', plot=p, height = 5, width = 9)
  
print(p)