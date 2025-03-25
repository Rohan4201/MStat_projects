library(tidyverse)
library(patchwork)


fara_df <- 
  read_csv("D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/farad.csv",
           na = c("", "NA", "NULL"))

to_be_removed <- (fara_df %>%
                    select(-(ends_with("20") | ends_with("20share"))) %>%
                    select(ends_with("10") | ends_with("10share")) %>%
                    colnames())[13:37]

fara_df_removed <- fara_df %>%
  select(-(ends_with("20") | ends_with("20share"))) %>%
  select(-all_of(to_be_removed))


rm(to_be_removed)

farad_both_years <- read_csv("D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/farad_both_years.csv")


us_export_data <- (readxl::read_xlsx(
"D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/State Agricultural Data/commodity_detail_by_state_cy.xlsx",
range = "Total exports!A3:V56"))[-c(1,2),] %>%
  rename("state" = `...1`) %>% 
  filter(state == "United States") %>% 
  mutate(`2007` = as.double(`2007`)) %>%
  pivot_longer(cols = -state,
               names_to = "years",
               values_to = "export_revenue_millions") 
###########################################################################


shp <- sf::st_read("D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/shapefiles/US/Counties/cb_2019_us_county_500k.shp")







###########################################################################################


################ Fonts

sysfonts::font_add_google(name = "Ubuntu", family = "ubuntu")
sysfonts::font_add_google(name = "Noto Serif", family = "noto")
sysfonts::font_add_google("IBM Plex Serif", "Regular 400")

fam <- "noto"
ubun <- "ubuntu"
plex_serif <- "Regular 400"

showtext::showtext_auto()

###########################################
#############################################################################################
##############################################################################################

sel_pov <- fara_df_removed %>%
  mutate(STATEFP = str_sub(CensusTract, 1, 2),
         COUNTYFP = str_sub(CensusTract, 3, 5),
         TRACTCE = str_sub(CensusTract, 6, 11)) %>%
  filter(!(State %in% c("Alaska", "Hawaii"))) %>%
  group_by(STATEFP, COUNTYFP) %>%
  summarise(county_med_pov_rate = median(PovertyRate, na.rm = T)) %>%
  ungroup() %>%
  left_join(fara_df_removed %>%
              mutate(STATEFP = str_sub(CensusTract, 1, 2),
                     COUNTYFP = str_sub(CensusTract, 3, 5)) %>%
              select(STATEFP, COUNTYFP, County, State) %>%
              distinct(),
            by = c("STATEFP", "COUNTYFP"))


#########  Oglala Lakota County (South Dakota) has three tracts which do not have values for 
######## corresponding poverty rates


med_pov_plot <- ggplot()+
  geom_sf(data = shp %>%
            right_join(sel_pov, by = c("STATEFP", "COUNTYFP")),
          aes(geometry = geometry, fill = county_med_pov_rate))+
  scale_fill_distiller(name = "Median Poverty Rate", palette = "RdBu" ,
                       breaks = scales::pretty_breaks(n = 7))+
  labs(caption = "Source: Calculated from USDA Food Access Reasearch Atlas Data",
       title = "US: Countywise Median Poverty Rate in 2019")+
  theme_void()+
  coord_sf()+
  theme(legend.position = "right",
        legend.text = element_text(size = 23, family = ubun),
        legend.title =  element_text(size = 23, family = ubun),
        plot.title = element_text(size = 35, family = plex_serif, face="bold", hjust = 0.5),
        plot.caption = element_text(size = 22.5, family = ubun, hjust = 0))

ggsave(med_pov_plot, 
       filename = "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Codes/Initial plots/med_pov_plot.jpg",
       width = 22, height = 13, units = "cm")

rm(list= c("med_pov_plot", "sel_pov"))
###########################################
#############################################################################################
##############################################################################################


sel_snap_share <- fara_df_removed %>%
  mutate(STATEFP = str_sub(CensusTract, 1, 2),
         COUNTYFP = str_sub(CensusTract, 3, 5),
         TRACTCE = str_sub(CensusTract, 6, 11)) %>%
  filter(!(State %in% c("Alaska", "Hawaii"))) %>%
  group_by(STATEFP, COUNTYFP) %>%
  summarise(county_med_snap_share = median(lasnaphalfshare, na.rm = T)) %>%
  ungroup() %>%
  left_join(fara_df_removed %>%
              mutate(STATEFP = str_sub(CensusTract, 1, 2),
                     COUNTYFP = str_sub(CensusTract, 3, 5)) %>%
              select(STATEFP, COUNTYFP, County, State) %>%
              distinct(),
            by = c("STATEFP", "COUNTYFP"))


snap_share_plot <- ggplot()+
  geom_sf(data = shp %>%
            right_join(sel_snap_share, by = c("STATEFP", "COUNTYFP")),
          aes(geometry = geometry, fill = county_med_snap_share))+
  scale_fill_distiller(name = "Median share", palette = "RdBu" ,
                       breaks = scales::pretty_breaks(n = 7))+
  labs(caption = "Source: Calculated from USDA Food Access Reasearch Atlas Data",
       title = "US: Countywise median percentage of population receiving SNAP benefits in 2019")+
  theme_void()+
  coord_sf()+
  theme(legend.position = "right",
        legend.text = element_text(size = 23, family = ubun),
        legend.title =  element_text(size = 23, family = ubun),
        plot.title = element_text(size = 35, family = plex_serif, face="bold", hjust = 0.5),
        plot.caption = element_text(size = 22.5, family = ubun, hjust = 0))

ggsave(snap_share_plot, 
       filename = "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Initial plots/snap_share_plot.jpg",
       width = 22, height = 13, units = "cm")

rm(sel_snap_share, snap_share_plot)


##############################

farad_both_years %>%
  filter(lalowihalfshare_2019 > 90) %>%
  select(State,County, Pop2010, 
         LILATracts_halfAnd10_2019, LILATracts_halfAnd10_2015,
         LILATracts_Vehicle_2019, LILATracts_Vehicle_2015,
         lalowihalfshare_2019, lalowihalfshare_2015) %>%
  arrange(desc(lalowihalfshare_2019)) %>%
  View()



pov_rate_lalowihalf_2019 <- farad_both_years %>%
  filter(!is.na(PovertyRate_2019))%>%
  ggplot()+
  geom_boxplot(
    aes(x = cut_width(PovertyRate_2019, 10, boundary = 0), y = lalowihalfshare_2019,
        fill = cut_width(PovertyRate_2019, 10, boundary = 0)),
    varwidth=T, alpha = 0.5
  )+
  labs(x = "Poverty Rate",y = "Share of Population")+
  annotate("text",fontface="bold",hjust = 0,vjust=0.5,
           x=0,y=120,size=8.5, family = plex_serif,
           label="Year: 2019")+
  ggthemes::scale_fill_pander()+
#  annotate("text",
#           x=10,y=102,hjust=1,vjust=0.5,size=3.5,
#           label="Only 2 urban tracts both in North\nLouisiana region and belonging\nto LILA category")+
#  annotate("curve",
#           x=8.4,y=97,xend=8.85,yend=90,curvature=0.2,
#           arrow = arrow(length = unit(0.02, "npc")))+
  annotate("rect",
           xmin = 0.3, xmax = 5.5, ymin = 0, ymax = 105, alpha = 0.16)+
  scale_y_continuous(breaks=seq(0,100,20))+
  theme_minimal()+
  theme(legend.position="none",
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        plot.margin = unit(c(0,0,0,0),"npc"),
        axis.title = element_text(face="bold", size = 20, family = ubun),
        axis.text = element_text(face="bold", size = 17, family = ubun))




pov_rate_lalowihalf_2015 <- farad_both_years %>%
  filter(!is.na(PovertyRate_2015))%>%
  ggplot()+
  geom_boxplot(
    aes(x = cut_width(PovertyRate_2015, 10, boundary = 0), y = 100*lalowihalfshare_2015,
        fill = cut_width(PovertyRate_2015, 10, boundary = 0)),
    varwidth=T, alpha = 0.5
  )+
  labs(x = "Poverty Rate",y = "Share of Population")+
  annotate("text",fontface="bold",hjust = 0,vjust=0.5,
           x=0,y=120,size=8.5, family = plex_serif,
           label="Year: 2015")+
  ggthemes::scale_fill_pander()+
  #  annotate("text",
  #           x=10,y=102,hjust=1,vjust=0.5,size=3.5,
  #           label="Only 2 urban tracts both in North\nLouisiana region and belonging\nto LILA category")+
  #  annotate("curve",
  #           x=8.4,y=97,xend=8.85,yend=90,curvature=0.2,
  #           arrow = arrow(length = unit(0.02, "npc")))+
  annotate("rect",
           xmin = 0.3, xmax = 5.5, ymin = 0, ymax = 105, alpha = 0.16)+
  scale_y_continuous(breaks=seq(0,100,20))+
  theme_minimal()+
  theme(legend.position="none",
        panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        plot.margin = unit(c(0,0,0,0),"npc"),
        axis.title = element_text(face="bold", size = 20, family = ubun),
        axis.text = element_text(face="bold", size = 17, family = ubun))


ggsave(pov_rate_lalowihalf_2019/pov_rate_lalowihalf_2015 +
         plot_annotation(title = "Percentage of people in Low Income Low Access regions (half mile measure)",
                         theme = theme(
                           plot.title = element_text(face="bold", size = 35, family = plex_serif,
                                                     hjust = 0.5)
                         )),
       filename = "D:/Rohan/Maths/MStat/Semester 1/Regression techniques/Project/Initial plots/pov_rate_lalowihalf_boxplot.jpg",
       width = 25, height = 16, units = "cm")


rm(pov_rate_lalowihalf_2015, pov_rate_lalowihalf_2019)
























