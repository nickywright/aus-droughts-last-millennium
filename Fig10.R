## ---------------------------------------------------------------------------
##
## Script name: Code to replicate analyses in Falster et al (2024) HESS
##
## Purpose of script: Re-create Figure 10
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

# 1. annual-total timeseries of area-mean Murray-Darling Basin rainfall, for each CESM LME ensemble member
# 2. radiative forcings for the CESM1 Last Millennium Ensemble 

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
library(readr)
library(patchwork)
library(readxl)

# =============================================================================
# read in Murray-Darling Basin area-mean precipitation (from the CESM LME)
# =============================================================================

prec <- read.csv("pmip3-droughts/data/CESM-LME_MDB-area-mean-precip.csv") %>%
  filter(year <= 2000)

# ---------------------
# LM: anomalies relative to the entire period
# ---------------------

prec.anoms <- prec

for(i in 1:(ncol(prec.anoms))) {
  
  if(colnames(prec.anoms[i]) == "year") {
    next
  }
  
  prec.anoms[, i] <- prec.anoms[ ,i] - mean(prec.anoms[ ,i])
  
}

# =============================================================================
# read in CESM LME radiative forcings from Schmidt et al 2012 
# "Climate forcing reconstructions for use in PMIP simulations of the Last Millennium (v1.1)"
# =============================================================================

# https://gmd.copernicus.org/articles/5/185/2012/gmd-5-185-2012.pdf

{
  f_lulc <- read_excel("pmip3-droughts/data/PMIP_LM_radforc_v1.1.xlsx", sheet = 1) %>%
    select(year, val = PEA)
  
  f_ssi <- read_excel("pmip3-droughts/data/PMIP_LM_radforc_v1.1.xlsx", sheet = 2) %>%
    select(year, val = VSK)
  
  f_ghg <- read_excel("pmip3-droughts/data/PMIP_LM_radforc_v1.1.xlsx", sheet = 3) %>%
    select(year, val = WMGHG)
  
  f_volc <- read_excel("pmip3-droughts/data/PMIP_LM_radforc_v1.1.xlsx", sheet = 4) %>%
    select(year, val = GRA)
}

# ---------------------
# join them into a single df
# ---------------------

all_forcings <- data.frame(year = f_ghg$year,
                           lulc = f_lulc$val,
                           ssi = f_ssi$val,
                           ghg = f_ghg$val,
                           volc = f_volc$val) %>%
  mutate(total = lulc+ssi+ghg+volc) %>%
  pivot_longer(-year, values_to = "val", names_to = "forcing")

# =============================================================================
# identify 2S2E droughts
# =============================================================================

prec.2s2e.lm <- prec.anoms

for(i in 1:(ncol(prec.2s2e.lm))) {
  
  if(colnames(prec.2s2e.lm[i]) == "year") {
    next
  }
  
  rowcount <- 1
  
  while(rowcount <= nrow(prec.2s2e.lm)) {
    
    # end test
    if(rowcount == 1151) {
      
      if(prec.2s2e.lm[rowcount-1 ,i] == "yes" & prec.anoms[rowcount ,i] < 0) {
        
        prec.2s2e.lm[rowcount ,i] <- "yes"
        
      } else {
        
        prec.2s2e.lm[rowcount ,i] <- "no"
      }
      
      rowcount = rowcount+1
    }
    
    # another end test
    
    if(prec.anoms[rowcount ,i] < 0 & prec.anoms[rowcount+1 ,i] < 0 & rowcount == 1150) {
      
      prec.2s2e.lm[rowcount ,i] <- "yes"
      prec.2s2e.lm[rowcount+1 ,i] <- "yes"
      
      rowcount <- rowcount+1
    }
    
    # check for first drought.2S2E year
    
    if(rowcount < nrow(prec.anoms)) {
      if(prec.anoms[rowcount ,i] < 0 & prec.anoms[rowcount+1 ,i] < 0) {
        
        prec.2s2e.lm[rowcount ,i] <- "yes"
        
        # ok we've started a 2S2E drought: let's see when it finishes
        
        done <- "no"
        k <- 1
        
        while(done == "no") {
          
          if(prec.anoms[rowcount+k ,i] > 0 & prec.anoms[rowcount+1+k ,i] > 0) {
            
            prec.2s2e.lm[rowcount+k ,i] <- "no"
            prec.2s2e.lm[rowcount+1+k ,i] <- "no"
            
            done <- "yes"
            
          } else {
            
            prec.2s2e.lm[rowcount+k ,i] <- "yes"
            k = k+1
          }
          
          if(rowcount+1+k > 1151) {
            
            prec.2s2e.lm[rowcount+k ,i] <- "yes"
            done <- "yes"
          }
        }
        
      } else {
        
        prec.2s2e.lm[rowcount, i] <- "no" 
        k = 1
      }
      
      # so we've identified a drought.2S2E (or found that this isn't the start of one). Let's update the counter and go again
      
      rowcount <- rowcount+k
      
      rm(k)
    }
  }
  
}

prec.2s2e.long <- pivot_longer(prec.2s2e.lm, -year, names_to = "ensmem", values_to = "drought")

prec.lm.long <- pivot_longer(prec, -year, names_to = "ensmem", values_to = "prec") %>%
  mutate(forcing = gsub('[[:digit:]]+', '', ensmem)) %>%
  mutate(ensnum = parse_number(ensmem)) %>%
  left_join(prec.2s2e.long, by = c("year", "ensmem"))

# =============================================================================
# count how many ensemble members show drought in each year
# =============================================================================

prec.lm.long <- prec.lm.long %>%
  mutate(ensmem_in_drought.single = NA,
         ensmem_in_drought.all = NA)

for(year in unique(prec.lm.long$year)) {
  
  for(ff in unique(prec.lm.long$forcing)) {
    
    # count up droughts for each sub-ensemble 
    
    thischunk <- prec.lm.long[which(prec.lm.long$year == year & prec.lm.long$forcing == ff), ]
    
    droughtcount <- nrow(thischunk[which(thischunk$drought == "yes"), ])
    
    prec.lm.long$ensmem_in_drought.single[which(prec.lm.long$year == year & prec.lm.long$forcing == ff)] <- droughtcount
    
  }
  
  # count up droughts for the full LME
  
  thischunk <- prec.lm.long[which(prec.lm.long$year == year), ]
  
  droughtcount <- nrow(thischunk[which(thischunk$drought == "yes"), ])
  
  prec.lm.long$ensmem_in_drought.all[which(prec.lm.long$year == year)] <- droughtcount
  
}

# =============================================================================
# make a version of that df, with just one entry per forcing
# =============================================================================

prec.lm.long.red <-  prec.lm.long %>%
  distinct(year, forcing, .keep_all = TRUE) %>%
  select(-c(ensmem, ensnum))

# =============================================================================
# preparation for plotting
# =============================================================================

drought.bc <- prec.lm.long.red %>%
  # set coordinates
  mutate(top = case_when(forcing == "ff" ~ 6,
                         forcing == "ghg" ~ 5,
                         forcing == "volc" ~ 4,
                         forcing == "orb" ~ 3,
                         forcing == "solar" ~ 2,
                         forcing == "lulc" ~ 1),
         bottom = top-1) %>%
  na.omit() %>%
  # Put NAs where there's no drought
  mutate(ensmem_in_drought.single = ifelse(ensmem_in_drought.single > 0, ensmem_in_drought.single, NA)) %>%
  mutate(ensmem_in_drought.all = ifelse(ensmem_in_drought.all > 0, ensmem_in_drought.all, NA)) 

y_label_df <- select(drought.bc, c(forcing, top)) %>%
  unique() %>%
  arrange(desc(top))

y_labels <- y_label_df$forcing
names(y_labels) <- y_label_df$top-0.5

# ---------------------
# scale the 'number of ensemble members in drought' for plotting
# ---------------------

drought.bc <- drought.bc %>%
  mutate(droughts.scaled = case_when(forcing == "ff" ~ as.numeric(ensmem_in_drought.single/13),
                                     forcing == "orb" ~ as.numeric(ensmem_in_drought.single/3),
                                     forcing == "solar" ~ as.numeric(ensmem_in_drought.single/4),
                                     forcing == "volc" ~ as.numeric(ensmem_in_drought.single/4),
                                     forcing == "ghg" ~ as.numeric(ensmem_in_drought.single/3),
                                     forcing == "lulc" ~ as.numeric(ensmem_in_drought.single/3)))

# add scaling for 'all'

drought.bc <- drought.bc %>%
  mutate(droughts.scaled.all = as.numeric(ensmem_in_drought.all/length(unique(prec.lm.long$ensmem))))


# ---------------------
# a version of the df to draw boxes around the forcings
# ---------------------

forcing_bounds <- group_by(drought.bc, forcing) %>% 
  unique() %>%
  summarise(top = max(top),
            bottom = min(bottom))

# =============================================================================
# 'barcode' plot showing years in drought for each ensemble member
# =============================================================================

barcode.plot.subens <- ggplot() +
  ## full forcing
  geom_rect(data = na.omit(filter(drought.bc, forcing == "ff")), 
            aes(xmin = year, xmax = year+1, ymin = top-1, ymax = top, colour = droughts.scaled)) +
  
  # well-mixed GHGs
  geom_rect(data = na.omit(filter(drought.bc, forcing == "ghg")), 
            aes(xmin = year, xmax = year+1, ymin = top-1, ymax = top, colour = droughts.scaled)) +
  
  # volcanic
  geom_rect(data = na.omit(filter(drought.bc, forcing == "volc")), 
            aes(xmin = year, xmax = year+1, ymin = top-1, ymax = top, colour = droughts.scaled)) +
  
  # orbital
  geom_rect(data = na.omit(filter(drought.bc, forcing == "orb")), 
            aes(xmin = year, xmax = year+1, ymin = top-1, ymax = top, colour = droughts.scaled)) +
  # solar
  geom_rect(data = na.omit(filter(drought.bc, forcing == "solar")), 
            aes(xmin = year, xmax = year+1, ymin = top-1, ymax = top, colour = droughts.scaled)) +
  
  # LULC
  geom_rect(data = na.omit(filter(drought.bc, forcing == "lulc")), 
            aes(xmin = year, xmax = year+1, ymin = top-1, ymax = top, colour = droughts.scaled)) +


  # rectangles to group the forcings types
  geom_rect(data = forcing_bounds, aes(group = forcing, xmin = -Inf, xmax = Inf, ymin = bottom, ymax = top), 
            fill = NA, colour = "black", size = 1) +
  # axis bounds and labels
  scale_x_continuous(limits = c(850, 2001), expand = c(0,0), breaks = seq(900, 2000, 100), minor_breaks = seq(850, 2000, 50)) +
  scale_y_continuous(breaks = seq(0.5, 5.5, 1), labels = rev(y_labels), expand = c(0, 0)) +
  # set a more obvious colour scale
  scale_colour_viridis(option = "inferno", direction = -1, limits = c(0,1)) +
  # labels
  labs(x = "Year (CE)", colour = "Proportion of ensemble\nmembers in drought") +
  theme_bw() +
  theme(panel.grid = element_blank())

# =============================================================================
# one barcode, with the entire ensemble
# =============================================================================

barcode.plot.fullens <- ggplot() +
  ## full forcing
  geom_rect(data = drought.bc, 
            aes(xmin = year, xmax = year+1, ymin = 0, ymax = 1, colour = droughts.scaled.all)) +

  # axis bounds and labels (I have NO idea why i need to reverse the y labels but there you go)
  scale_x_continuous(limits = c(850, 2001), expand = c(0,0), breaks = seq(900, 2000, 100), minor_breaks = seq(850, 2000, 50)) +
  scale_y_continuous(breaks = seq(0, 1, 1), expand = c(0, 0)) +
  # set a more obvious colour scale
  scale_colour_viridis(option = "inferno",direction = -1, limits = c(0,1)) +
  # labels
  labs(x = "Year (CE)", colour = "Proportion of ensemble\nmembers in drought") +
  theme_bw() +
  theme(panel.grid = element_blank()) +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.title.x = element_blank())

# =============================================================================
# now add the radiative forcing timeseries
# =============================================================================

forcings <- ggplot() +
  # all except volcanic
  geom_line(data = filter(all_forcings, forcing != "volc" & forcing != "total"), aes(x = year, y = val, group = forcing, colour = forcing)) +
  # volcanic only
  geom_line(data = filter(all_forcings, forcing == "volc"), aes(x = year, y = (val/10)+2, group = forcing, colour = forcing)) +
  # summed
  #Geom_line(data = filter(all_forcings, forcing == "total"), aes(x = year, y = val, colour = "black")) + 
  # dual y-axis scaling
  scale_y_continuous(name = "Radiative forcing (W/"~m^2~") all but volcanic", 
                     sec.axis = sec_axis(~(.-2)*10,
                                         name = "Radiative forcing (W/"~m^2~") volcanic only")) +
  # make it match the other plots
  scale_x_continuous(limits = c(850, 2001), expand = c(0,0), breaks = seq(900, 2000, 100), minor_breaks = seq(850, 2000, 50)) +
  # colours
  scale_colour_viridis_d(direction = -1) +
  theme_bw()

# =============================================================================
# stack the plots
# =============================================================================

forcings + barcode.plot.fullens + barcode.plot.subens + plot_layout(ncol = 1, heights = c(2, 1, 6), guides = "collect")

