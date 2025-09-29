Host-Side Scalars Support {#dev_guide_host_side_scalars}
========================================================

oneDNN supports a special memory object for host-side scalar values. Creating
such an object is lightweight and does not require specifying an engine.

To create a host-side scalar memory object, first create a memory descriptor
with the scalar data type. Then, use this descriptor and a scalar value to
create the memory object. The scalar value is copied into the memory object,
so its lifetime does not need to be managed by the user.

Using the C++ API:
```C
float alpha = 1.0f;

// Create a memory object for a host scalar of type float
dnnl::memory alpha_mem(memory::desc::host_scalar(memory::data_type::f32), alpha);
```

Using the C API:
```C
float alpha = 1.0f;

// Create a memory descriptor for a host scalar of type float
dnnl_memory_desc_t alpha_md;
dnnl_memory_desc_create_host_scalar(&alpha_md, dnnl_f32);

// Create a memory object for the scalar
dnnl_memory_t alpha_mem;
dnnl_memory_create_host_scalar(&alpha_mem, alpha_md, &alpha);
```

The memory object can then be used in primitives just like any other memory object.

If at any point the user needs to access or update the scalar value,
they can do so using @ref dnnl_memory_get_host_scalar_value and
@ref dnnl_memory_set_host_scalar_value, or using the C++ API:
```C
// Get the scalar value from the memory object
float alpha_value = alpha_mem.get_host_scalar_value();

float new_alpha_value = 2.0f;
// Update the scalar value in the memory object
alpha_mem.set_host_scalar_value(&new_alpha_value);
```