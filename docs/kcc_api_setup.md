# KCC Transcript API Setup (Blank Folder)

## Project Structure
Create the standard folders:
```bash
mkdir -p config data docs scripts
```

## Environment Configuration
Create `config/kcc.env` with these exact values:
```dotenv
# KCC API config
KCC_API_BASE_URL=https://api.data.gov.in/resource/cef25fe2-9231-4128-8aec-2c948fedd43f
KCC_API_KEY=579b464db66ec23bdd000001b27ef70e3ab7476a7675e04561a0aeb2
KCC_DEFAULT_FORMAT=csv
KCC_DEFAULT_YEAR=2024
KCC_DEFAULT_LIMIT=10000
KCC_PAGE_MODE=1
KCC_DEDUP=0
```

## API Usage
**Purpose**
- The API provides Kisan Call Centre (KCC) transcript records from data.gov.in.
- Each record represents a farmer query and the corresponding response.

**Endpoint**
```
https://api.data.gov.in/resource/cef25fe2-9231-4128-8aec-2c948fedd43f
```

**Authentication**
- Supply `api-key` as a query parameter.

**Output Formats**
- Use `format=csv` for bulk exports.
- Use `format=json` for programmatic parsing.

**Pagination**
- Use `offset` and `limit` to page through records.
- Example: `offset=0&limit=10000`, then increment offset by limit.

**Filters**
Use query filters to narrow results:
- `filters[year]`
- `filters[month]`
- `filters[StateName]`

**Example Requests**
```text
https://api.data.gov.in/resource/cef25fe2-9231-4128-8aec-2c948fedd43f?api-key=YOUR_API_KEY&format=csv&offset=0&limit=10000&filters[year]=2024
```

```text
https://api.data.gov.in/resource/cef25fe2-9231-4128-8aec-2c948fedd43f?api-key=YOUR_API_KEY&format=json&offset=0&limit=10000&filters[year]=2024&filters[StateName]=UTTAR%20PRADESH
```

**Returned Fields**
- `StateName, DistrictName, BlockName, Season, Sector, Category, Crop, QueryType, QueryText, KccAns, CreatedOn, Year, Month`

## Data Handling Rules
- Do not modify raw CSV files.
- Write derived outputs to `data/derived/`.
- Document new analysis steps in `docs/`.
