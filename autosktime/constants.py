UNIVARIATE_FORECAST = 1
UNIVARIATE_ENDOGENOUS_FORECAST = UNIVARIATE_FORECAST
UNIVARIATE_EXOGENOUS_FORECAST = 2

FORECAST_TASK = [UNIVARIATE_FORECAST, UNIVARIATE_EXOGENOUS_FORECAST]

TASK_TYPES = FORECAST_TASK

TASK_TYPES_TO_STRING = {
    UNIVARIATE_FORECAST: 'forecast.univariate.endogenous',
    UNIVARIATE_EXOGENOUS_FORECAST: 'forecast.univariate.exogenous'
}
STRING_TO_TASK_TYPES = {value: key for key, value in TASK_TYPES_TO_STRING.items()}

HANDLES_UNIVARIATE = 'cap:handles-univariate'
HANDLES_MULTIVARIATE = 'cap:handles-multivariate'
HANDLES_CATEGORICAL = 'cap:handles-categorical'
IGNORES_EXOGENOUS_X = 'ignores-exogenous-X'
SUPPORTED_INDEX_TYPES = 'supported-index-types'

MAXINT = 2 ** 31 - 1
