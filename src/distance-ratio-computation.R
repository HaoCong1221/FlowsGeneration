# Title     : Distance ratio
# Objective : Based on travel survey and simulation
# Created by: Yuan Liao
# Created on: 2022-04-24

library(dplyr)

# Travel survey based ratio
df.total <- read.csv('dbs/distance_ratio_data.csv')
df.total <- df.total[df.total$distance_network >= df.total$distance, ]
df.total$diff <- df.total$distance_network/df.total$distance

# Get ratio of Sweden and the Netherlands
ratio.data.extraction <- function(region){
  df <- df.total[df.total$region==region, ]
  df$dg <- cut(df$distance, breaks = unlist(lapply(seq(-1, 4, 5/30), function(x){10^(x)})))

  df_stats <- df %>%
    group_by(dg)  %>%
    summarise(distance = median(distance),
              ratio = median(diff),
              lower_ratio = quantile(diff, 0.25),
              upper_ratio = quantile(diff, 0.75))

  df_stats <- subset(df_stats, select = -dg)
  df_stats$country <- region
  return(df_stats)
}

df.ratio.data <- rbind(ratio.data.extraction('sweden'),
                       ratio.data.extraction('netherlands'))
write.csv(df.ratio.data, "results/ratio_data.csv", row.names = FALSE)

# Simulation-based ratio
df.total <- read.csv('dbs/distance_ratio_simulation.csv')
df.total$gp <- cut(df.total$distance, breaks = unlist(lapply(seq(-1, 3, 4/30), function(x){10^(x)})))

df_stats <- df.total %>%
  group_by(gp)  %>%
  summarise(distance = median(distance),
            ratio = median(diff),
            lower_ratio = quantile(diff, 0.25),
            upper_ratio = quantile(diff, 0.75))
df_stats <- subset(df_stats, select = -gp)
write.csv(df_stats, "results/ratio_simulation.csv", row.names = FALSE)