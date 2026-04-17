import cutlass
import cutlass.cute as cute

@cute.jit
def derive_MMA_Layout(L_dst, L_XY, bits: cutlass.Constexpr):
    L_dst_dtype = cute.recast_layout(bits, 1, L_dst)
    Layout = cute.composition(cute.right_inverse(L_XY), L_dst_dtype)
    return Layout

@cute.jit
def sm75_ldsm():
    L_dst = cute.make_layout((32,(32,2)), stride=(32,(1,1024)))
    # L_mk (16,8):(8,1) happens to be identical as 2x1 block
    L_mk = cute.make_layout(((8,2),8), stride=((8,64),1))
    print(derive_MMA_Layout(L_dst, L_mk, 16))

@cute.jit
def sm80_ldsm():
    L_dst_ldsm4 = cute.make_layout((32,(32,4)), stride=(32,(1,1024)))
    # L_mk is not simple (16,16):(16,1), but a 2x2 block
    L_mk = cute.make_layout(((8,2),(8,2)), stride=((8,128),(1,64)))
    print(derive_MMA_Layout(L_dst_ldsm4, L_mk, 16))
    # L_nk is not simple (8,16):(16,1), but a 1x2 block
    L_dst_ldsm2 = cute.make_layout((32,(32,2)), stride=(32,(1,1024)))
    L_nk = cute.make_layout((8,(8,2)), stride=(8,(1,64)))
    print(derive_MMA_Layout(L_dst_ldsm2, L_nk, 16))
                            

if __name__ == "__main__":
    sm75_ldsm()
    sm80_ldsm()