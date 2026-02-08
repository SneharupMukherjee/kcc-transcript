# KCC Data Notes

## Source
Data.gov.in API resource:
- Resource ID: `cef25fe2-9231-4128-8aec-2c948fedd43f`
- Title: Kisan Call Centre (KCC) - Transcripts of farmers queries & answers

## Fields (CSV)
`StateName, DistrictName, BlockName, Season, Sector, Category, Crop, QueryType, QueryText, KccAns, CreatedOn, Year, Month`

## Typical Content
- `QueryText` is often in English or mixed language.
- `KccAns` is often in Hindi or Punjabi.
- `CreatedOn` is an ISO-like timestamp.

## API Notes
- Use filters like `filters[year]`, `filters[month]`, `filters[StateName]`.
- Pagination uses `offset` and `limit`.
