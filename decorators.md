# Decorators

The python decorators are purely for debugging purposes. They provide no additional network functionality other than to provide debugging information to the user. This debugging information is present in log levels such as `INFO2`.

## Current Decorator List

### Combinators

| Decorator | Status |
| ------------- | ------------- |
| Serial | &check; |
| Parallel | &check; |
| FanOut | &check; |
| FanInSum | &check; |
| FanInConcat | &check; |
| shape_dependent | &cross; |

### Layers

| Decorator | Status |
| ------------- | ------------- |
| Dense | &check; |
| Conv | &check; |
| Flatten | &check; |
| Identity | &check; |
| Reshape | &check; |
| MaxPool | &check; |
| SumPool | &check; |
| AvgPool | &check; |
| Dropout | &check; |
| BatchNorm | &cross; |

### Activation Functions

| Decorator | Status |
| ------------- | ------------- |
| Activation Functions | &check; |