const std = @import("std");
const assert = std.debug.assert;
const testing = std.testing;
const expectEqual = testing.expectEqual;

const c = @cImport({
    @cInclude("ggml/ggml.h");
});

const DataType = enum(c.ggml_type) {
    i8    = c.GGML_TYPE_I8,
    i16   = c.GGML_TYPE_I16,
    i32   = c.GGML_TYPE_I32,
    f16   = c.GGML_TYPE_F16,
    f32   = c.GGML_TYPE_F32,
};

// available tensor operations:
const TensorOp = enum(u32) {
    none,

    dup,
    add,
    sub,
    mul,
    div,
    sqr,
    sqrt,
    sum,
    mean,
    repeat,
    abs,
    sign,
    neg,
    step,
    relu,
    gelu,
    norm, // normalize

    mul_mat,

    scale,
    cpy,
    reshape,
    view,
    permute,
    transpose,
    get_rows,
    diag_mask_inf,
    softmax,
    rope,

    conv1d_1s,
    conv1d_2s,

    flash_attn,
    flash_ff,
};

const Tensor = extern struct {
    dtype: DataType,

    n_dims: i32,
    ne: [max_dims]i32,     // number of elements
    nb: [max_dims]usize,   // stride in bytes:
                           // nb[0] = @sizeOf(type)
                           // nb[1] = nb[0]   * ne[0] + padding
                           // nb[i] = nb[i-1] * ne[i-1]

    // compute data
    op: TensorOp,

    is_param: bool,

    grad: *Tensor,
    src0: *Tensor,
    src1: *Tensor,
    opt: [max_opt]*Tensor,

    // thread scheduling
    n_tasks: i32,

    // performance
    perf_runs: i32,
    perf_cycles: i64,
    perf_time_us: i64,

    data: *u8,
    padding: [8]u8,

    const max_opt = c.GGML_MAX_OPT;
    const max_dims = c.GGML_MAX_DIMS;

    comptime {
        // if (@sizeOf(usize) == 4) {
        //     assert(@sizeOf(Tensor) == 152);
        // } else {
        //     assert(@sizeOf(Tensor) == 152);
        // }
    }

    pub fn setZero(tensor: *Tensor) *Tensor {
        return @ptrCast(*Tensor, c.ggml_tensor_set_zero(@ptrCast(c.ggml_tensor, tensor)).?);
    }

    pub fn set(tensor: *Tensor, value: anytype) *Tensor {
        switch (@TypeOf(value)) {
            i32 => c.ggml_set_i32(@ptrCast(c.ggml_tensor, tensor), value),
            f32 => c.ggml_set_f32(@ptrCast(c.ggml_tensor, tensor), value),
            else => @compileError("Unsupported type"),
        }

        return tensor;
    }

    pub fn get1d(tensor: *const Tensor, comptime dtype: type, i: i32) dtype {
        switch (dtype) {
            i32 => return @intCast(i32, c.ggml_get_i32_1d(tensor, i)),
            f32 => return @intCast(f32, c.ggml_get_f32_1d(tensor, i)),
            else => @compileError("Unsupported type"),
        }
    }

    pub fn set1d(tensor: *Tensor, i: i32, value: anytype) *Tensor {
        switch (@TypeOf(value)) {
            i32 => c.ggml_set_i32_1d(tensor, i, value),
            f32 => c.ggml_set_f32_1d(tensor, i, value),
            else => @compileError("Unsupported type"),
        }

        return tensor;
    }

    pub fn getData(tensor: *const Tensor, comptime dtype: type) []u8 {
        switch (dtype) {
            i32 => {
                assert(tensor.dtype == DataType.i32);
                return @ptrCast(*i32, tensor.data);
            },
            f32 => {
                assert(tensor.dtype == DataType.f32);
                return @ptrCast(*f32, tensor.data);
            },
            else => @compileError("Unsupported type"),
        }
    }
};

const ContextParams = struct {
    mem_size: usize,
    mem_buffer: ?[]u8 = null,
};

const Context = struct {
    ggml: *c.ggml_context,

    pub fn init(params: ContextParams) !Context {
        assert(params.mem_size > 0);
        assert(params.mem_buffer == null or params.mem_buffer.?.len >= params.mem_size);

        var ggml = c.ggml_init(.{
            .mem_size   = params.mem_size,
            .mem_buffer = if (params.mem_buffer) |b| b.ptr else c.NULL,
        });
        
        if (ggml == null) {
            return error.OutOfMemory;
        }

        return Context{ .ggml = ggml.? };
    }

    pub fn deinit(self: *Context) void {
        c.ggml_free(self.ggml);
    }

    pub fn printObjects(self: *Context) void {
        c.ggml_print_objects(self.ggml);
    }

    //
    // Tensor creation
    //

    pub fn tensor(self: *Context, dtype: DataType, ne: []i32) !*Tensor {
        var t = c.ggml_new_tensor(self.ggml, @enumToInt(dtype), @intCast(c_int, ne.size), ne.ptr);
        if (t == null) {
            return error.OutOfMemory;
        }
        return @ptrCast(Tensor, t);
    }

    pub fn tensor1d(self: *Context, dtype: DataType, ne0: i32) !*Tensor {
        var t = c.ggml_new_tensor_1d(self.ggml, @enumToInt(dtype), @intCast(c_int, ne0));
        if (t == null) {
            return error.OutOfMemory;
        }
        return @ptrCast(*Tensor, t);
    }

    pub fn tensor2d(self: *Context, dtype: DataType, ne0: i32, ne1: i32) !*Tensor {
        var t = c.ggml_new_tensor_2d(self.ggml, @enumToInt(dtype), @intCast(c_int, ne0), @intCast(c_int, ne1));
        if (t == null) {
            return error.OutOfMemory;
        }
        return @ptrCast(*Tensor, t);
    }

    pub fn tensor3d(self: *Context, dtype: DataType, ne0: i32, ne1: i32, ne2: i32) !*Tensor {
        var t = c.ggml_new_tensor_3d(self.ggml, @enumToInt(dtype), @intCast(c_int, ne0), @intCast(c_int, ne1), @intCast(c_int, ne2));
        if (t == null) {
            return error.OutOfMemory;
        }
        return @ptrCast(*Tensor, t);
    }

    pub fn tensor4d(self: *Context, dtype: DataType, ne0: i32, ne1: i32, ne2: i32, ne3: i32) !*Tensor {
        var t = c.ggml_new_tensor_4d(self.ggml, @enumToInt(dtype), 
            @intCast(c_int, ne0), 
            @intCast(c_int, ne1),
            @intCast(c_int, ne2), 
            @intCast(c_int, ne3));
        if (t == null) {
            return error.OutOfMemory;
        }
        return @ptrCast(*Tensor, t);
    }

    pub fn one(self: *Context, value: anytype) !*Tensor {
        var t = switch (@TypeOf(value)) {
            i32 => c.ggml_new_i32(self.ggml, value),
            f32 => c.ggml_new_f32(self.ggml, value),
            else => @compileError("Unsupported type"),
        };

        if (t == null) {
            return error.OutOfMemory;
        }
        c.ggml_tensor_set_f32_1d(t, 0, value);
        return @ptrCast(*Tensor, t);
    }

    //
    // Operations on tensors with backpropagation
    //

    pub fn dup(self: *Context, t1: *Tensor, t2: *Tensor) !*Tensor {
        var t = c.ggml_dup(self.ggml, t1, t2);
        if (t == null) {
            return error.OutOfMemory;
        }
        return @ptrCast(*Tensor, t);
    }

    pub fn add(self: *Context, t1: *Tensor, t2: *Tensor) !*Tensor {
        var t = c.ggml_add(self.ggml, t1, t2);
        if (t == null) {
            return error.OutOfMemory;
        }
        return @ptrCast(*Tensor, t);
    }

    pub fn sub(self: *Context, t1: *Tensor, t2: *Tensor) !*Tensor {
        var t = c.ggml_sub(self.ggml, t1, t2);
        if (t == null) {
            return error.OutOfMemory;
        }
        return @ptrCast(*Tensor, t);
    }

    pub fn mul(self: *Context, t1: *Tensor, t2: *Tensor) !*Tensor {
        var t = c.ggml_mul(self.ggml, t1, t2);
        if (t == null) {
            return error.OutOfMemory;
        }
        return @ptrCast(*Tensor, t);
    }

    pub fn sqr(self: *Context, a: *Tensor) !*Tensor {
        var t = c.ggml_sqr(self.ggml, a);
        if (t == null) {
            return error.OutOfMemory;
        }
        return @ptrCast(*Tensor, t);
    }

    pub fn sqrt(self: *Context, a: *Tensor) !*Tensor {
        var t = c.ggml_sqrt(self.ggml, a);
        if (t == null) {
            return error.OutOfMemory;
        }
        return @ptrCast(*Tensor, t);
    }
};

// -----------------------------------------------------------------------------
// Units tests
// -----------------------------------------------------------------------------

test "test0: basic add functionality" {
    const allocator = std.testing.allocator;

    const mem_size = 128 * 1024 * 1024;

    var mem_buffer = try allocator.alloc(u8, mem_size);
    defer allocator.free(mem_buffer);

    var ctx0 = try Context.init(.{ .mem_size = mem_size, .mem_buffer = mem_buffer });
    defer ctx0.deinit();

    var t1 = try ctx0.tensor1d(.f32, 10);
    var t2 = try ctx0.tensor2d(.i16, 10, 20);
    var t3 = try ctx0.tensor3d(.i32, 10, 20, 30);

    try expectEqual(t1.*.n_dims, 1);
    try expectEqual(t1.*.ne[0] , 10);
    try expectEqual(t1.*.nb[1] , 10 * @sizeOf(f32));

    try expectEqual(t2.*.n_dims, 2);
    try expectEqual(t2.*.ne[0] , 10);
    try expectEqual(t2.*.ne[1] , 20);
    try expectEqual(t2.*.nb[1] , 10 * @sizeOf(i16));
    try expectEqual(t2.*.nb[2] , 10 * 20 * @sizeOf(i16));

    try expectEqual(t3.*.n_dims, 3);
    try expectEqual(t3.*.ne[0] , 10);
    try expectEqual(t3.*.ne[1] , 20);
    try expectEqual(t3.*.ne[2] , 30);
    try expectEqual(t3.*.nb[1] , 10 * @sizeOf(i32));
    try expectEqual(t3.*.nb[2] , 10 * 20 * @sizeOf(i32));
    try expectEqual(t3.*.nb[3] , 10 * 20 * 30 * @sizeOf(i32));

    ctx0.printObjects();
}

fn is_close(a: f32, b: f32, epsilon: f32) bool {
    return @abs(a - b) < epsilon;
}

test "test3" {
    var params = ContextParams{
        .mem_size   = 1024 * 1024 * 1024,
        .mem_buffer = null,
    };

    var opt_params = c.ggml_opt_default_params(c.GGML_OPT_LBFGS);
    //struct ggml_opt_params opt_params = ggml_opt_default_params(GGML_OPT_ADAM);

    opt_params.n_threads = 8;

    const NP = 1 << 12;
    const NF = 1 << 8;

    var ctx0 = Context.init(params);

    var F = ctx0.tensor2d(.f32, NF, NP);
    var l = ctx0.tensor1d(.f32, NP);

    // regularization weight
    var lambda = ctx0.one(1e-5);

    const RndGen = std.rand.DefaultPrng;
    var rnd = RndGen.init(0);

    var j: usize = 0;
    while (j < NP) : (j += 1) {
        const ll = if (j < NP / 2) 1.0 else -1.0;
        @ptrCast(*f32, l.data)[j] = ll;

        var i: usize = 0;
        while (i < NF): (i += 1) {
            @ptrCast(*f32, F.data)[j * NF + i] = 
                ((if ((if (ll > 0 and i < NF/2) 1.0 else ll < 0) and i >= NF / 2) 1.0 else 0.0) 
                + (rnd.random.float(f32) - 0.5) * 0.1) / (0.5 * NF);
        }
    }

    {
        // initial guess
        var x = ctx0.tensor1d(.f32, NF);
        x.set(0.0);

        ggml_set_param(ctx0, x);

        // f = sum[(fj*x - l)^2]/n + lambda*|x^2|
        var f =
            ggml_add(ctx0,
                    ggml_div(ctx0,
                        ggml_sum(ctx0,
                            ggml_sqr(ctx0,
                                ggml_sub(ctx0,
                                    ggml_mul_mat(ctx0, F, x),
                                    l)
                                )
                            ),
                        ggml_new_f32(ctx0, NP)
                        ),
                    ggml_mul(ctx0,
                        ggml_sum(ctx0, ggml_sqr(ctx0, x)),
                        lambda)
                    );

        enum ggml_opt_result res = ggml_opt(NULL, opt_params, f);

        assert(res == GGML_OPT_OK);

        // print results
        for (int i = 0; i < 16; i++) {
            printf("x[%3d] = %g\n", i, ((float *)x->data)[i]);
        }
        printf("...\n");
        for (int i = NF - 16; i < NF; i++) {
            printf("x[%3d] = %g\n", i, ((float *)x->data)[i]);
        }
        printf("\n");

        for (int i = 0; i < NF; ++i) {
            if (i < NF/2) {
                assert(is_close(((float *)x->data)[i],  1.0f, 1e-2f));
            } else {
                assert(is_close(((float *)x->data)[i], -1.0f, 1e-2f));
            }
        }
    }

    ggml_free(ctx0);

    return 0;
}
