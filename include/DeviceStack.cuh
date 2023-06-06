#ifndef _DEVICE_STACK_CUH_
#define _DEVICE_STACK_CUH_

template <typename T, int size>
class DeviceStack
{
    private:
        T stack[size];
        int top;

    public:
        __device__ DeviceStack()
        {
            top = -1;
        }

        __device__ void clear()
        {
            top = -1;
        }

        __device__ void push(T value)
        {
            if (top < size - 1)
            {
                top++;
                stack[top] = value;
            }
        }

        __device__ T pop()
        {
            if (top >= 0)
            {
                T value = stack[top];
                top--;
                return value;
            }
            return T();
        }

        __device__ T front()
        {
            return stack[top];
        }

        __device__ bool is_full() const
        {
            return top >= size - 1;
        }

        __device__ bool is_empty() const
        {
            return top == -1;
        }
};

#endif