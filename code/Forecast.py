# Databricks notebook source
# MAGIC %md
# MAGIC # Install Pycaret

# COMMAND ----------

!pip install pycaret[full]
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Load region_margin data

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC REFRESH TABLE region_margin;
# MAGIC select * from region_margin;

# COMMAND ----------

region_margins_sqldf=_sqldf
region_margins_df = region_margins_sqldf.select('margin','product_line','region','month').toPandas()
import pandas as pd
region_margins_df['month'] = pd.to_datetime(region_margins_df['month'])

display(region_margins_df.info())

# COMMAND ----------

# MAGIC %md
# MAGIC # TS Forecasting Experiment - Setup

# COMMAND ----------

from pycaret.time_series import *
exp = TSForecastingExperiment()
data = region_margins_df[['month','margin']]
s = setup(data=data,target='margin', fh = 3, session_id = 123)
exp.setup(data=data,target='margin', fh=3, session_id=1)  


# COMMAND ----------

# MAGIC %md
# MAGIC # Check Data Stats

# COMMAND ----------

s.check_stats()

# COMMAND ----------

# MAGIC %md
# MAGIC # Compare Experiment Models

# COMMAND ----------

best = exp.compare_models()


# COMMAND ----------

# plot forecast
plot_model(best, plot = 'forecast')

# COMMAND ----------

# plot forecast for 36 months in future
plot_model(best, plot = 'forecast', data_kwargs = {'fh' : 36})


# COMMAND ----------

# residuals plot
plot_model(best, plot = 'residuals')


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Prediction

# COMMAND ----------

holdout_pred = predict_model(best)


# COMMAND ----------

holdout_pred.head()


# COMMAND ----------

predict_model(best, fh = 5)


# COMMAND ----------


