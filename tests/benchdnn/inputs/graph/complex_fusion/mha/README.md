# SDPA Test Files

To facilitate the teting of SDPA functionalities, we unify the tensor IDs and
operation IDs used in these test files as follows:

## Tensor IDs

| Tensor       | ID      | Note                       |
| :==:         | :==:    | :==:                       |
| Query        | 1       | Input query                |
| Key          | 2       | Input key                  |
| Value        | 3       | Input value                |
| Scale        | 4       | Input scale                |
| Mask         | 5       | Input mask                 |
| Output       | 6       | Output                     |
| Score        | 7       | Internal output of QK/BMM1 |
| Scaled score | 8       | Internal output of scale   |
| Masked score | 9       | Internal output of mask    |
| Probs        | 10      | Internal output of softmax |


## Operation IDs

| Operation    | ID      | Note                            |
| :==:         | :==:    | :==:                            |
| QK/BMM1      | 101     | MatMul for query and key        |
| Scale        | 102     | Scale the output of QK/BMM1     |
| Mask         | 103     | Mask the output of scale        |
| Softmax      | 104     | Softmax over the output of mask |
| VS/BMM2      | 105     | MatMul for probs and value      |
