# Mausam APIs (IMD + APISetu)

## Sources
- IMD API PDF: https://mausam.imd.gov.in/imd_latest/contents/api.pdf
- APISetu collection page: https://directory.apisetu.gov.in/api-collection/mausam

## Access Requirements
IMD Mausam APIs require IP whitelisting. Access is granted after submitting the IMD API request form and receiving approval. Use the official form and contact listed below.
- Request form: https://city.imd.gov.in/citywx/api_request.php
- Contact email: kavita.navria@imd.gov.in

## IMD API List (from PDF, de-duplicated)
1. City Weather forecast for 7 days `✗` (connection reset by peer)
Endpoint: https://city.imd.gov.in/api/cityweather.php
Sample: https://city.imd.gov.in/api/cityweather.php?id=42182

2. City Weather forecast for 7 days with latitude/longitude `✗` (IP needs to be whitelisted)
Endpoint: https://city.imd.gov.in/api/cityweather_loc.php
Sample: https://city.imd.gov.in/api/cityweather_loc.php?id=42182

3. Current Weather `✗` (IP/Domain needs to be whitelisted)
Endpoint: https://mausam.imd.gov.in/api/current_wx_api.php
Sample: https://mausam.imd.gov.in/api/current_wx_api.php?id=42182

4. District Wise Nowcast `✗` (IP needs to be whitelisted)
Endpoint: https://mausam.imd.gov.in/api/nowcast_district_api.php
Sample: https://mausam.imd.gov.in/api/nowcast_district_api.php?id=5

5. District wise Rainfall `✗` (IP needs to be whitelisted)
Endpoint: https://mausam.imd.gov.in/api/districtwise_rainfall_api.php

6. District wise Warning `✗` (IP needs to be whitelisted)
Endpoint: https://mausam.imd.gov.in/api/warnings_district_api.php
Sample: https://mausam.imd.gov.in/api/warnings_district_api.php?id=1

7. Station Wise Nowcast `✗` (IP needs to be whitelisted)
Endpoint: https://mausam.imd.gov.in/api/nowcastapi.php
Sample: https://mausam.imd.gov.in/api/nowcastapi.php?id=Jaipur%20AP

8. State wise Rainfall `✗` (IP needs to be whitelisted)
Endpoint: https://mausam.imd.gov.in/api/statewise_rainfall_api.php

9. RSS Feeds
Endpoint: https://mausam.imd.gov.in/imd_latest/contents/dist_nowcast_rss.php

10. AWS/ARG Data `✗` (IP/Domain needs to be whitelisted)
Endpoint: https://city.imd.gov.in/api/aws_data_api.php

11. River Basin (Quantitative Precipitation Forecast) `✗` (IP needs to be whitelisted)
Endpoint: https://mausam.imd.gov.in/api/basin_qpf_api.php

12. Port Warning `✗` (IP needs to be whitelisted)
Endpoint: https://mausam.imd.gov.in/api/port_wx_api.php

13. Sea Area Bulletin `✗` (IP needs to be whitelisted)
Endpoint: https://mausam.imd.gov.in/api/seaarea_bulletin_api.php

14. Coastal Area Bulletin `✗` (IP needs to be whitelisted)
Endpoint: https://mausam.imd.gov.in/api/coastal_bulletin_api.php

15. Subdivisional APIs
- https://mausam.imd.gov.in/api/api_5d_subdivisional_rf.php `✗` (IP needs to be whitelisted)
- https://mausam.imd.gov.in/api/api_5d_statewisedistricts_rf_forecast.php `✗` (IP needs to be whitelisted)
- https://mausam.imd.gov.in/api/api_subDivisionWiseWarning.php `✗` (IP needs to be whitelisted)
- https://mausam.imd.gov.in/api/subdivisionwise_rainfall_api.php `✗` (IP needs to be whitelisted)

## APISetu "mausam" Collection
The APISetu collection page renders the API list client-side. The API list was not present in the static HTML at the time of review, so no additional endpoints are listed here.
