"""
spark_job.py — Chicago Crimes COVID-19 CRUD con Spark YARN
Ejecutar con:
  spark-submit \
    --master yarn \
    --deploy-mode client \
    --executor-memory 6g \
    --executor-cores 2 \
    --num-executors 2 \
    --driver-memory 4g \
    --conf spark.sql.shuffle.partitions=16 \
    /root/spark_job.py
"""
import time
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    LongType, DoubleType
)

PROJECT_ID    = 'my-first-project-492901'
DATASET_ID    = 'chicago_crimes_results'
GCS_BUCKET    = 'gs://big-data-proyecto-parcial/raw'
YEARS         = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

# ── SparkSession — YARN provisto por spark-submit ─────────────────────────────
spark = (
    SparkSession.builder
    .appName('ChicagoCrimes-COVID-CRUD-YARN')
    .config('spark.ui.showConsoleProgress', 'true')
    .getOrCreate()
)
sc = spark.sparkContext
print(f'\n{"="*60}')
print(f'Spark {spark.version}')
print(f'Master:   {sc.master}')
print(f'App ID:   {sc.applicationId}')
print(f'Cores:    {sc.defaultParallelism}')
print(f'{"="*60}\n')

# ── Helper BigQuery ───────────────────────────────────────────────────────────
def to_bigquery(spark_df, table_name, if_exists='replace'):
    pdf = spark_df.toPandas()
    pdf.to_gbq(
        destination_table=f'{DATASET_ID}.{table_name}',
        project_id=PROJECT_ID,
        if_exists=if_exists,
        progress_bar=False,
    )
    print(f'  → BigQuery: {DATASET_ID}.{table_name}  ({len(pdf):,} filas)')

# ── Schema ────────────────────────────────────────────────────────────────────
schema = StructType([
    StructField('unique_key',           LongType(),    True),
    StructField('case_number',          StringType(),  True),
    StructField('date',                 StringType(),  True),
    StructField('block',                StringType(),  True),
    StructField('iucr',                 StringType(),  True),
    StructField('primary_type',         StringType(),  True),
    StructField('description',          StringType(),  True),
    StructField('location_description', StringType(),  True),
    StructField('arrest',               StringType(),  True),
    StructField('domestic',             StringType(),  True),
    StructField('beat',                 IntegerType(), True),
    StructField('district',             IntegerType(), True),
    StructField('ward',                 IntegerType(), True),
    StructField('community_area',       IntegerType(), True),
    StructField('fbi_code',             StringType(),  True),
    StructField('x_coordinate',         LongType(),    True),
    StructField('y_coordinate',         LongType(),    True),
    StructField('year',                 IntegerType(), True),
    StructField('updated_on',           StringType(),  True),
    StructField('latitude',             DoubleType(),  True),
    StructField('longitude',            DoubleType(),  True),
    StructField('location',             StringType(),  True),
])

# ── Carga desde GCS — conector Hadoop nativo ──────────────────────────────────
t0 = time.time()
files = [f'{GCS_BUCKET}/Chicago_Crimes_{y}.csv' for y in YEARS]

era_expr = (
    F.when(F.col('year') <= 2019, 'PRE')
     .when(F.col('year') <= 2022, 'DURANTE')
     .otherwise('POST')
)

df = (
    spark.read
         .option('header', 'true')
         .option('nullValue', '')
         .schema(schema)
         .csv(files)
         .withColumn('date',       F.to_timestamp('date'))
         .withColumn('updated_on', F.to_timestamp('updated_on'))
         .withColumn('arrest',     F.lower(F.col('arrest')) == 'true')
         .withColumn('domestic',   F.lower(F.col('domestic')) == 'true')
         .withColumn('covid_era',  era_expr)
)
df.cache()
total = df.count()

# Capturar métricas de executors DESPUÉS de que YARN los registre
import time as _time
_time.sleep(5)
total_cores = spark.sparkContext.defaultParallelism
executor_ids = spark.sparkContext._jvm.scala.collection.JavaConverters \
    .seqAsJavaListConverter(
        spark.sparkContext._jsc.sc().executorAllocationManager()
    ).asJava() if False else []

# Usar REST API de Spark para contar executors
try:
    import urllib.request, json
    port = spark.sparkContext.uiWebUrl.split(':')[-1]
    url  = f'http://localhost:{port}/api/v1/applications/{spark.sparkContext.applicationId}/executors'
    with urllib.request.urlopen(url, timeout=5) as r:
        executors = json.loads(r.read())
    active = [e for e in executors if not e.get('isBlacklisted', False) and e['id'] != 'driver']
    hosts  = list({e['hostPort'].split(':')[0] for e in active})
except Exception as ex:
    active, hosts = [], [str(ex)]

print(f'Fuente:              GCS nativo — {GCS_BUCKET}/')
print(f'Registros:           {total:,}  ({time.time()-t0:.1f}s)')
print(f'Años:                {YEARS}')
print(f'Cores en paralelo:   {total_cores}')
print(f'Executors activos:   {len(active)}')
print(f'Hosts de executors:  {hosts}\n')
df.createOrReplaceTempView('crimes')

# ── CRUD 1 — CREATE: Arrestos por año y era ───────────────────────────────────
print('--- CRUD 1: CREATE arrests_by_year ---')
arrests_by_year = spark.sql("""
    SELECT
        year, covid_era,
        COUNT(*)                                 AS total_crimes,
        SUM(CAST(arrest AS INT))                 AS total_arrests,
        ROUND(AVG(CAST(arrest AS INT))*100, 2)   AS arrest_rate_pct
    FROM crimes
    WHERE year IS NOT NULL
    GROUP BY year, covid_era
    ORDER BY year
""")
arrests_by_year.show(30)
to_bigquery(arrests_by_year, 'arrests_by_year')

# ── CRUD 2 — READ: Crímenes por community area ────────────────────────────────
print('--- CRUD 2: READ crimes_by_district ---')
crimes_by_area = spark.sql("""
    SELECT
        community_area AS district, covid_era,
        COUNT(*)                                 AS total_crimes,
        SUM(CAST(arrest AS INT))                 AS total_arrests,
        ROUND(AVG(CAST(arrest AS INT))*100, 2)   AS arrest_rate_pct,
        COUNT(DISTINCT primary_type)             AS distinct_crime_types,
        SUM(CAST(domestic AS INT))               AS domestic_incidents
    FROM crimes
    WHERE community_area IS NOT NULL
    GROUP BY community_area, covid_era
    ORDER BY community_area, covid_era
""")
print('Top 5 áreas por era:')
for era in ['PRE', 'DURANTE', 'POST']:
    crimes_by_area.filter(F.col('covid_era') == era) \
        .orderBy(F.col('total_crimes').desc()) \
        .select('district', 'total_crimes', 'arrest_rate_pct') \
        .show(5)
to_bigquery(crimes_by_area, 'crimes_by_district')

# ── CRUD 3 — READ: Tendencia mensual ─────────────────────────────────────────
print('--- CRUD 3: READ monthly_trend ---')
monthly_trend = spark.sql("""
    SELECT
        year, covid_era,
        MONTH(date)                              AS month,
        COUNT(*)                                 AS total_crimes,
        SUM(CAST(arrest AS INT))                 AS arrests,
        ROUND(AVG(CAST(arrest AS INT))*100, 2)   AS arrest_rate_pct
    FROM crimes
    WHERE date IS NOT NULL AND year IS NOT NULL
    GROUP BY year, covid_era, MONTH(date)
    ORDER BY year, month
""")
print(f'Puntos temporales: {monthly_trend.count()}')
to_bigquery(monthly_trend, 'monthly_trend')

# ── CRUD 4 — UPDATE: is_weekend + day_name ───────────────────────────────────
print('--- CRUD 4: UPDATE weekend_analysis ---')
df_enriched = (
    df.withColumn('is_weekend', F.dayofweek('date').isin([1, 7]))
      .withColumn('day_name',   F.date_format('date', 'EEEE'))
)
df_enriched.createOrReplaceTempView('crimes')

weekend_analysis = spark.sql("""
    SELECT
        covid_era, is_weekend,
        COUNT(*)                                              AS total_crimes,
        SUM(CAST(arrest AS INT))                             AS total_arrests,
        ROUND(AVG(CAST(arrest AS INT))*100, 2)               AS arrest_rate_pct,
        ROUND(COUNT(*)*100.0 / SUM(COUNT(*)) OVER(PARTITION BY covid_era), 2) AS pct_of_era_total
    FROM crimes
    WHERE is_weekend IS NOT NULL
    GROUP BY covid_era, is_weekend
    ORDER BY covid_era, is_weekend DESC
""")
weekend_analysis.show()
to_bigquery(weekend_analysis, 'weekend_analysis')

# ── CRUD 5 — DELETE: Filtrar registros inválidos ──────────────────────────────
print('--- CRUD 5: DELETE registros inválidos ---')
invalid_by_era = (
    df_enriched
    .filter(F.col('primary_type').isNull() | (F.trim(F.col('primary_type')) == '') | F.col('unique_key').isNull())
    .groupBy('covid_era').count()
    .withColumnRenamed('count', 'invalid_records')
)
total_by_era = (
    df_enriched.groupBy('covid_era').count()
               .withColumnRenamed('count', 'total')
)
report = (
    total_by_era.join(invalid_by_era, on='covid_era', how='left')
                .fillna(0)
                .withColumn('pct_invalid', F.round(F.col('invalid_records')/F.col('total')*100, 4))
                .orderBy('covid_era')
)
print(f'Total registros: {total:,}')
report.show()

df_valid = df_enriched.filter(
    F.col('primary_type').isNotNull() &
    (F.trim(F.col('primary_type')) != '') &
    F.col('unique_key').isNotNull()
)
print(f'Registros válidos: {df_valid.count():,}')

print(f'\n{"="*60}')
print('Todos los CRUDs completados. Resultados en BigQuery.')
print(f'Tiempo total: {time.time()-t0:.1f}s')
print(f'{"="*60}')
spark.stop()
