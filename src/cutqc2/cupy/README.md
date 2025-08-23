The `main.cu` file is not technically needed, but serves to test the kernel
in a standalone manner.

```
nvcc -o main main.cu -lcudart
./main
```