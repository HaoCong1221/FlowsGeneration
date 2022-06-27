# Title     : Metropolitan 1-km grids align with Pop SSPs
# Objective : Extract pop ssps from .tif and form 1-km grid zones
# Created by: Yuan Liao
# Created on: 2022-06-13

# load the raster and rgdal libraries
library(raster)
library(maptools)
library(rgdal)
library(glue)
library(stars)
library(rgeos)
library(sp)
library(geosphere)
library(parallelsugar)


region <- 'sweden'
for (yr in c(2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100)){
  for (ssp in seq(1, 5)) {
    path_shp <- 'dbs/{region}/zones/DeSO/DeSO_VGR.shp'
    path_rst <- "D:/navigate_tt/POP/popdynamics-1-km-downscaled-pop-base-year-projection-ssp-2000-2100-rev01-proj-ssp{ssp}-geotiff/SSP{ssp}_1km/ssp{ssp}_total_{yr}.tif"

    # import the vector boundary
    crop_extent <- readOGR(glue(path_shp))

    # open raster layer
    pop <- raster(glue(path_rst))

    # reproject
    crop_extent.reproj <- spTransform(crop_extent, crs(pop))
    # get the population mass center of a given zone: multi_polygon[, id] from a given pop raster
    # crop to get intersected raster part
    pop_crop <- crop(pop, crop_extent.reproj)

    # convert raster grid to polygon
    r.to.poly <- rasterToPolygons(pop_crop)
    r.to.poly$zone <- seq_along(r.to.poly)

    # clip the converted raster polygon by polygon
    r.to.poly.clipped <- crop(r.to.poly, crop_extent.reproj, snap = 'out')

    r.to.poly <- r.to.poly[r.to.poly$zone %in% r.to.poly.clipped$zone, ]
    names(r.to.poly) <- c('pop', 'zone')
    r.to.poly <- spTransform(r.to.poly, crs(crop_extent))
    shapefile(r.to.poly, filename=glue('dbs/{region}/ssps/{region}_vg_ssp{ssp}_yr{yr}.shp'))

    # add area information
    r.to.poly <- read_sf(glue('dbs/{region}/ssps/{region}_vg_ssp{ssp}_yr{yr}.shp'))
    r.to.poly$Area <- st_area(r.to.poly)
    r.to.poly$Area <- as.numeric(r.to.poly$Area) / 10^6 # km^2
    # add r assuming they are square grids (approximately true)
    r.to.poly$r <- r.to.poly$Area^0.5
    st_write(r.to.poly, glue('dbs/{region}/ssps/{region}_vg_ssp{ssp}_yr{yr}.shp'), append=FALSE)
    print(glue('{region} SSP-{ssp} for the year {yr} is done.'))
  }
}


