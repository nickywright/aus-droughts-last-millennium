## ---------------------------------------------------------------------------
##
## Script name: Code to replicate analyses in Falster et al (2024) HESS
##
## Purpose of script: Re-create Figure 1 - observed Murray-Darling Basin precipitation anomalies
##
## Script author: Georgina Falster
##
## Date create/updated: 2024-02-19
##
## Email: georgina.falster@anu.edu.au (institutional) / georgina.falster@gmail.com (permanent)
##
## Citation: https://doi.org/10.5194/egusphere-2023-1398

## ----------------------------------------------------------------------------
##
## Please cite the paper if using or re-purposing this code.
## 
## ----------------------------------------------------------------------------

# =============================================================================
# Notes for running this script
# =============================================================================

# This script requires the following inputs:

# 1. annual-total timeseries of observed area-mean Murray-Darling Basin rainfall

# Please edit all filepaths as necessary to where you have these data stored

# =============================================================================
# set display options
# =============================================================================

options(scipen = 10, digits = 4)

# =============================================================================
# load packages
# =============================================================================

library(magrittr)
library(tidyverse)
library(viridis)

# =============================================================================
# read in Murray-Darling Basin area-mean precipitation (from AGCD)
# =============================================================================

# I had to do this in two parts - there may now be a continuous version of AWAP

awap <- read.csv("data/AGCDv1_annual_precip_MDB_1900-2020.csv") %>%
  rename(prec = precip) %>%
  mutate(year = as.numeric(substr(date, 1, 4))) %>%
  select(-date) %>%
  filter(year < 2020)

awap1 <- read.csv("data/AGCDv1_annual_precip_MDB_2020-2022.csv") %>%
  rename(prec = precip) %>%
  mutate(year = as.numeric(substr(date, 1, 4))) %>%
  select(-date)

awap <- rbind(awap, awap1)

rm(awap1)

# ---------------------
# anomalies relative to the entire period
# ---------------------

prec.mean <- mean(awap$prec)

awap$prec.anom <- awap$prec-prec.mean

rm(prec.mean)

# =============================================================================
# identify 2S2E droughts
# =============================================================================

awap$drought.2S2E <- NA

rowcount <- 1

while(rowcount <= nrow(awap)) {
  
  # end test
  if(rowcount == 123) {
    
    if(awap$drought.2S2E[rowcount-1] == "yes" & awap$prec.anom[rowcount] < 0) {
      
      awap$drought.2S2E[rowcount] <- "yes"
      
    } else {
      
      awap$drought.2S2E[rowcount] <- "no"
    }
    
    rowcount = rowcount+1
  }
  
  # another end test
  
  if(awap$prec.anom[rowcount] < 0 & awap$prec.anom[rowcount +1] < 0 & rowcount == 122) {
    
    awap$drought.2S2E[rowcount] <- "yes"
    awap$drought.2S2E[rowcount+1] <- "yes"
    
    rowcount <- rowcount+1
  }
  
  # check for first drought.2S2E year
  
  if(rowcount < nrow(awap)) {
    if(awap$prec.anom[rowcount] < 0 & awap$prec.anom[rowcount +1] < 0) {
      
      awap$drought.2S2E[rowcount] <- "yes"
      
      # ok we've started a 2S2E drought: let's see when it finishes
      
      done <- "no"
      k <- 1
      
      while(done == "no") {
        
        if(awap$prec.anom[rowcount+k] > 0 & awap$prec.anom[rowcount+1+k] > 0) {
          
          awap$drought.2S2E[rowcount+k] <- "no"
          awap$drought.2S2E[rowcount+1+k] <- "no"
          
          done <- "yes"
          
        } else {
          
          awap$drought.2S2E[rowcount+k] <- "yes"
          k = k+1
        }
        
        if(rowcount+1+k > 123) {
          
          awap$drought.2S2E[rowcount+k] <- "yes"
          done <- "yes"
        }
      }
      
    } else {
      
      awap$drought.2S2E[rowcount] <- "no" 
      k = 1
    }
    
    # so we've identified a drought.2S2E (or found that this isn't the start of one). Let's update the counter and go again
    
    rowcount <- rowcount+k
    
    rm(k)
  }
}

# =============================================================================
# Historical droughts (from Helman 2009)
# =============================================================================

# Available from https://silo.tips/download/droughts-in-the-murray-darling-basin-since-european-settlement

# Note: these are multi-year droughts only (from Table 1), with Millennium Drought adjusted to recent 'accepted' values

# only including droughts referred to in the text (Falster et al. 2024 HESS)

hist.droughts <- data.frame(name = c("Federation",  "WWII","Millennium", "Tinderbox"),
                            start = c(1895, 1935, 1997, 2017),
                            end = c(1903, 1945,2009, 2019))

hist.droughts$middle <- rowMeans(select(hist.droughts, -name))

# adjust Federation Drought so it shows up

hist.droughts$middle[which(hist.droughts$name == "Federation")] <- 1901.5

# =============================================================================
# Plot the results
# =============================================================================

# -------------------------------------------------------------
# some preparations
# -------------------------------------------------------------

awap <- mutate(awap, col_tag = ifelse(prec.anom < 0, "neg", "pos"))

col_vals <- c("neg" = "#8C510A", "pos" = "#01665E")

# -------------------------------------------------------------
# the plot
# -------------------------------------------------------------

prec_plot <- ggplot(data = awap) +
  # add windows showing historical droughts
  # added 1 so it goes to the end of the year
  geom_rect(data = hist.droughts, aes(xmin = start, xmax = end+1, ymin = 0, ymax = Inf), alpha = 0.4, fill = "darkorange3") + 
  # and the drought names
  geom_text(data = hist.droughts, aes(x = middle, y = 185, label = name), angle = 90, hjust = 0, size = 6, 
            colour = "darkorange4", fontface = "bold") +
  # the precip timeseries
  geom_col(aes(x = year, y = prec.anom, fill = col_tag), position = position_nudge(x = 0.5)) +
  # colours scaling
  scale_fill_manual(values = col_vals) +
  # show 2S2E droughts s windows on the bottom
  geom_rect(data = filter(awap, drought.2S2E == "yes"), aes(xmin = year, xmax = year+1, ymin = -Inf, ymax = 0), alpha = 0.4, fill = "grey") + 

  # sort out x axis
  scale_x_continuous(breaks = seq(1900, 2022, 10)) +
  scale_y_continuous(limits = c(-250, 350), breaks = seq(-250, 300, 50)) +
  coord_cartesian(xlim = c(1900, 2022), expand = FALSE) +
  # labels etc
  labs(x = "Year (CE)", y = "Annual precipitation anomaly (mm)\nover the Murray-Darling Basin") +
  guides(fill = "none") +
  # and finally themes
  theme_bw() +
  theme(axis.text.y = element_text(size = 10),
        axis.text.x = element_text(size = 12),
        axis.title = element_text(size = 12),
        )

