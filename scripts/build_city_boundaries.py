"""Extract reviewed city polygons from official 2025 Census TIGER/Line places.

Run from the repository root. State ZIPs are cached outside the repository;
only the 72 selected polygons and their provenance are saved under data/.
"""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import tempfile
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd
from pyproj import CRS, Transformer
import shapefile
from shapely.geometry import shape, mapping
from shapely.ops import transform

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(tempfile.gettempdir()) / 'spotter-tiger2025'
STATES = dict(zip(
    'AL AZ AR CA CT DC FL GA IL IN KY LA MD MA MI MS MO NV NM NY NC OH OK PA RI SC TN TX UT VA WI'.split(),
    '01 04 05 06 09 11 12 13 17 18 21 22 24 25 26 28 29 32 35 36 37 39 40 42 44 45 47 48 49 51 55'.split(),
))
ALIASES = {
    'Indianapolis': 'Indianapolis city (balance)',
    'Lexington': 'Lexington-Fayette',
    'Louisville': 'Louisville/Jefferson County metro government (balance)',
    'Nashville': 'Nashville-Davidson metropolitan government (balance)',
}


def read_state(state, layer="PLACE"):
    code = STATES[state]
    name = f'tl_2025_{code}_{layer.lower()}'
    url = f'https://www2.census.gov/geo/tiger/TIGER2025/{layer}/{name}.zip'
    path = CACHE / f'{name}.zip'
    if not path.exists():
        with urlopen(url, timeout=120) as response:
            content = response.read()
        temporary = path.with_suffix('.tmp')
        temporary.write_bytes(content)
        temporary.replace(path)
    with ZipFile(path) as archive:
        reader = shapefile.Reader(shp=BytesIO(archive.read(f'{name}.shp')),
                                  shx=BytesIO(archive.read(f'{name}.shx')),
                                  dbf=BytesIO(archive.read(f'{name}.dbf')), encoding='utf-8')
        projection = CRS.from_wkt(archive.read(f'{name}.prj').decode())
        records = list(reader.iterShapeRecords())
    print(f'{state}: {len(records)} place records', flush=True)
    return state, records, projection, url


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    cities = pd.read_csv(ROOT / 'data/city-coordinates.csv')
    with ThreadPoolExecutor(max_workers=4) as pool:
        states = {state: (records, projection, url) for state, records, projection, url
                  in pool.map(read_state, sorted(cities.state.unique()))}
    consolidated = {city: read_state(state, 'CONCITY')[1:] for city, state in
                    [('Indianapolis', 'IN'), ('Louisville', 'KY'), ('Nashville', 'TN')]}
    consolidated_names = {'Indianapolis': 'Indianapolis', 'Louisville': 'Louisville/Jefferson County',
                          'Nashville': 'Nashville-Davidson'}
    features = []
    for city in cities.itertuples(index=False):
        records, projection, url = states[city.state]
        name = ALIASES.get(city.city, city.city)
        if city.city in consolidated:
            records, projection, url = consolidated[city.city]
            name = consolidated_names[city.city]
        candidates = [r for r in records if r.record.as_dict()['NAME'] == name]
        if len(candidates) != 1:
            choices = [(r.record.as_dict()['NAME'], r.record.as_dict()['NAMELSAD']) for r in records
                       if city.city.split()[0].lower() in r.record.as_dict()['NAME'].lower()]
            raise ValueError(f'{city.city}, {city.state}: {len(candidates)} matches; candidates: {choices}')
        record = candidates[0]
        attrs = record.record.as_dict()
        transformer = Transformer.from_crs(projection, 'EPSG:4326', always_xy=True)
        geometry = transform(transformer.transform, shape(record.shape.__geo_interface__))
        if not geometry.is_valid:
            raise ValueError(f'Invalid source polygon: {city.city}')
        features.append({'type': 'Feature', 'id': city.city,
                         'properties': {'city': city.city, 'state': city.state, 'geoid': attrs['GEOID'],
                                        'census_name': attrs['NAMELSAD'], 'classfp': attrs['CLASSFP'],
                                        'boundary_date': '2025-01-01', 'source': url,
                                        'boundary_type': 'consolidated city' if city.city in consolidated else 'Census place'},
                         'geometry': mapping(geometry)})
    result = {'type': 'FeatureCollection', 'features': features}
    path = ROOT / 'data/city-boundaries-2025.geojson'
    path.write_text(json.dumps(result, separators=(',', ':')))
    print(f'Saved {len(features)} full-resolution polygons: {path.name} ({path.stat().st_size / 1e6:.2f} MB)')


if __name__ == '__main__':
    main()
