import numpy as np

from Quantizer import *


def testQuantization(
        verbose: list = [],
):
    """
    Basic test of the quantization
    """
    print("\ntestQuantization")
    vec2quantize = np.array([-0.08, -0.01, 0.08, 0.05])
    cntrSize = 8
    if VERBOSE_PRINT_SCREEN in verbose:
        print(f'vec2quantize={vec2quantize}\n')
    if VERBOSE_DEBUG in verbose:
        debugFile = open('../res/debug.txt', 'w')
        printf(debugFile, f'vec2quantize={vec2quantize}')

    # Test signed int grid
    grid = np.array(range(-2 ** (cntrSize - 1) + 1, 2 ** (cntrSize - 1), 1), dtype='int')
    [quantizedVec, scale, z] = quantize(vec=vec2quantize, grid=grid)
    dequantizedVec = dequantize(quantizedVec, scale, z)
    if VERBOSE_PRINT_SCREEN in verbose:
        print(
            f'grid={grid[:5]}...{grid[-5:]}\nquantizedVec={quantizedVec}, scale={scale}, z={z}\ndequantizedVec={dequantizedVec}\n')
    if VERBOSE_DEBUG in verbose:
        printf(debugFile,
               f'\ngrid={grid}\nquantizedVec={quantizedVec}\nscale={scale}, z={z}\ndequantizedVec={dequantizedVec}\n')

    # Test f2p_li_h2 grid
    grid = getAllValsFxp(
        fxpSettingStr='F2P_lr_h2',
        cntrSize=cntrSize,
        verbose=[],
        signed=True
    )
    [quantizedVec, scale, z] = quantize(vec=vec2quantize, grid=grid)
    dequantizedVec = dequantize(quantizedVec, scale, z)
    if VERBOSE_PRINT_SCREEN in verbose:
        print(
            f'grid={grid[:5]}...{grid[-5:]}\nquantizedVec={quantizedVec}, scale={scale}, z={z}\ndequantizedVec={dequantizedVec}\n')
    if VERBOSE_DEBUG in verbose:
        printf(debugFile,
               f'grid={grid}\nvec2quantize={vec2quantize}\nquantizedVec={quantizedVec}\nscale={scale}, z={z}\ndequantizedVec={dequantizedVec}\n')
        debugFile.close()

    print(
        "---------------------------------------------------------------------------------------------------------------------------------------------------------")


def testQuantization_INT16(
        verbose: list = [],
):
    """
    Basic test of the quantization
    """
    print("\ntestQuantization_INT16")
    #    vec2quantize = np.random.rand(20)
    #    vec2quantize = np.array([-8.395, 0.4668, 0.999,  0.3246, 0.8034, 0.5425, 0.3114, 0.6021, 0.3876, 0.1157,
    # 0.4667, 0.8732, 0.6634, 0.5993, 7.5224, 0.3644, 0.0337, 0.9877, 90.2061, 0.8461])

    vec2quantize = np.array([-100, -20.25, -5.75, 0, 0.5, 1, 5.5, 8.125, 70.825, 100])
    cntrSize = 16
    if VERBOSE_PRINT_SCREEN in verbose:
        print(f'vec2quantize={vec2quantize}')
    if VERBOSE_DEBUG in verbose:
        debugFile = open('../res/debug.txt', 'w')
        printf(debugFile, f'vec2quantize={vec2quantize}')

    # Test signed int grid
    grid = np.array(range(-2 ** (cntrSize - 1) + 1, 2 ** (cntrSize - 1), 1), dtype='int')
    [quantizedVec, scale, z] = quantize(vec=vec2quantize, grid=grid)
    dequantizedVec = dequantize(quantizedVec, scale, z)
    print(f'vec2quantize={vec2quantize}')
    if VERBOSE_PRINT_SCREEN in verbose:
        print(f'\ngrid={grid}\nquantizedVec={quantizedVec}, scale={scale}, z={z}\ndequantizedVec={dequantizedVec}\n')
    if VERBOSE_DEBUG in verbose:
        printf(debugFile,
               f'\ngrid={grid}\nquantizedVec={quantizedVec}\nscale={scale}, z={z}\ndequantizedVec={dequantizedVec}\n')

    # Test f2p_lr_h2 grid
    fxpSettingStr = 'F2P_lr_h2'
    sign = True
    grid = getAllValsFxp(
        fxpSettingStr='F2P_lr_h2',
        cntrSize=cntrSize,
        verbose=[],
        signed=sign
    )
    print(f'grid parameters:\n  fxpSettingsStr: {fxpSettingStr} | cntrSize: {cntrSize} | signed: {sign}')
    [quantizedVec, scale, z] = quantize(vec=vec2quantize, grid=grid)
    dequantizedVec = dequantize(quantizedVec, scale, z)
    print(f'vec2quantize={vec2quantize}')
    if VERBOSE_PRINT_SCREEN in verbose:
        print(
            f'grid={grid[:5]}...{grid[-5:]}\nquantizedVec={quantizedVec}, scale={scale}, z={z}\ndequantizedVec={dequantizedVec}\n')
    if VERBOSE_DEBUG in verbose:
        printf(debugFile,
               f'grid={grid}\nvec2quantize={vec2quantize}\nquantizedVec={quantizedVec}\nscale={scale}, z={z}\ndequantizedVec={dequantizedVec}')
        debugFile.close()

    # Test f2p_lr_h2 grid
    fxpSettingStr = 'F2P_lr_h2'
    sign = False
    grid = getAllValsFxp(
        fxpSettingStr='F2P_lr_h2',
        cntrSize=cntrSize,
        verbose=[],
        signed=sign
    )
    print(f'grid parameters:\n  fxpSettingsStr: {fxpSettingStr} | cntrSize: {cntrSize} | signed: {sign}')
    [quantizedVec, scale, z] = quantize(vec=vec2quantize, grid=grid)
    dequantizedVec = dequantize(quantizedVec, scale, z)
    print(f'vec2quantize={vec2quantize}')
    if VERBOSE_PRINT_SCREEN in verbose:
        print(
            f'grid={grid[:5]}...{grid[-5:]}\nquantizedVec={quantizedVec}, scale={scale}, z={z}\ndequantizedVec={dequantizedVec}\n')
    if VERBOSE_DEBUG in verbose:
        printf(debugFile,
               f'grid={grid}\nvec2quantize={vec2quantize}\nquantizedVec={quantizedVec}\nscale={scale}, z={z}\ndequantizedVec={dequantizedVec}')
        debugFile.close()

    # Test f2p_li_h2 grid
    fxpSettingStr = 'F2P_li_h2'
    sign = True
    grid = getAllValsFxp(
        fxpSettingStr='F2P_li_h2',
        cntrSize=cntrSize,
        verbose=[],
        signed=sign
    )
    print(f'grid parameters:\n  fxpSettingsStr: {fxpSettingStr} | cntrSize: {cntrSize} | signed: {sign}')
    [quantizedVec, scale, z] = quantize(vec=vec2quantize, grid=grid)
    dequantizedVec = dequantize(quantizedVec, scale, z)
    print(f'vec2quantize={vec2quantize}')
    if VERBOSE_PRINT_SCREEN in verbose:
        print(
            f'grid={grid[:5]}...{grid[-5:]}\nquantizedVec={quantizedVec}, scale={scale}, z={z}\ndequantizedVec={dequantizedVec}\n')
    if VERBOSE_DEBUG in verbose:
        printf(debugFile,
               f'grid={grid}\nvec2quantize={vec2quantize}\nquantizedVec={quantizedVec}\nscale={scale}, z={z}\ndequantizedVec={dequantizedVec}')
        debugFile.close()

    # Test f2p_li_h2 grid
    fxpSettingStr = 'F2P_li_h2'
    sign = False
    grid = getAllValsFxp(
        fxpSettingStr='F2P_li_h2',
        cntrSize=cntrSize,
        verbose=[],
        signed=sign
    )
    print(f'grid parameters:\n  fxpSettingsStr: {fxpSettingStr} | cntrSize: {cntrSize} | signed: {sign}')
    [quantizedVec, scale, z] = quantize(vec=vec2quantize, grid=grid)
    dequantizedVec = dequantize(quantizedVec, scale, z)
    print(f'vec2quantize={vec2quantize}')
    if VERBOSE_PRINT_SCREEN in verbose:
        print(f'grid={grid[:5]}...{grid[-5:]}\nquantizedVec={quantizedVec}, scale={scale}, z={z}\ndequantizedVec={dequantizedVec}\n')
    if VERBOSE_DEBUG in verbose:
        printf(debugFile,f'grid={grid}\nvec2quantize={vec2quantize}\nquantizedVec={quantizedVec}\nscale={scale}, z={z}\ndequantizedVec={dequantizedVec}')
        debugFile.close()

    # Test F2P_si_h2 grid
    fxpSettingStr = 'F2P_si_h2'
    sign = True
    grid = getAllValsFxp(
        fxpSettingStr=fxpSettingStr,
        cntrSize=cntrSize,
        verbose=[],
        signed=sign
    )
    print(f'grid parameters:\n  fxpSettingsStr: {fxpSettingStr} | cntrSize: {cntrSize} | signed: {sign}')
    [quantizedVec, scale, z] = quantize(vec=vec2quantize, grid=grid)
    dequantizedVec = dequantize(quantizedVec, scale, z)
    print(f'vec2quantize={vec2quantize}')
    if VERBOSE_PRINT_SCREEN in verbose:
        print(
            f'grid={grid[:5]}...{grid[-5:]}\nquantizedVec={quantizedVec}, scale={scale}, z={z}\ndequantizedVec={dequantizedVec}\n')
    if VERBOSE_DEBUG in verbose:
        printf(debugFile,
               f'grid={grid}\nvec2quantize={vec2quantize}\nquantizedVec={quantizedVec}\nscale={scale}, z={z}\ndequantizedVec={dequantizedVec}')
        debugFile.close()

    # Test F2P_si_h2 grid
    fxpSettingStr = 'F2P_si_h2'
    sign = False
    grid = getAllValsFxp(
        fxpSettingStr=fxpSettingStr,
        cntrSize=cntrSize,
        verbose=[],
        signed=sign
    )
    print(f'grid parameters:\n  fxpSettingsStr: {fxpSettingStr} | cntrSize: {cntrSize} | signed: {sign}')
    [quantizedVec, scale, z] = quantize(vec=vec2quantize, grid=grid)
    dequantizedVec = dequantize(quantizedVec, scale, z)
    print(f'vec2quantize={vec2quantize}')
    if VERBOSE_PRINT_SCREEN in verbose:
        print(f'grid={grid[:5]}...{grid[-5:]}\nquantizedVec={quantizedVec}, scale={scale}, z={z}\ndequantizedVec={dequantizedVec}')
    if VERBOSE_DEBUG in verbose:
        printf(debugFile,f'grid={grid}\nvec2quantize={vec2quantize}\nquantizedVec={quantizedVec}\nscale={scale}, z={z}\ndequantizedVec={dequantizedVec}')
        debugFile.close()

    # todo - fix this test FP_e5 exsit??
    # # Test FP_e5 grid
    # fxpSettingStr = 'FP_e5'
    # grid = getAllValsFxp(
    #     fxpSettingStr=fxpSettingStr,
    #     cntrSize=cntrSize,
    #     verbose=[],
    #     signed=True
    # )
    # print(f'grid parameters:\n  fxpSettingsStr: {fxpSettingStr} | cntrSize: {cntrSize} | signed: True')
    # [quantizedVec, scale, z] = quantize(vec=vec2quantize, grid=grid)
    # dequantizedVec = dequantize(quantizedVec, scale, z)
    # if VERBOSE_PRINT_SCREEN in verbose:
    #     print(f'grid={grid[:5]}...{grid[-5:]}\nquantizedVec={quantizedVec}, scale={scale}, z={z}\ndequantizedVec={dequantizedVec}')
    # if VERBOSE_DEBUG in verbose:
    #     printf(debugFile,
    #                 f'grid={grid}\nvec2quantize={vec2quantize}\nquantizedVec={quantizedVec}\nscale={scale}, z={z}\ndequantizedVec={dequantizedVec}')
    #     debugFile.close()

    # todo - fix this test FP_e2 exsit??
    # # Test FP_e2 grid
    # fxpSettingStr = 'FP_e2'
    # grid = getAllValsFxp(
    #     fxpSettingStr=fxpSettingStr,
    #     cntrSize=cntrSize,
    #     verbose=[],
    #     signed=True
    # )
    # print(f'grid parameters:\n  fxpSettingsStr: {fxpSettingStr} | cntrSize: {cntrSize} | signed: True')
    # [quantizedVec, scale, z] = quantize(vec=vec2quantize, grid=grid)
    # dequantizedVec = dequantize(quantizedVec, scale, z)
    # if VERBOSE_PRINT_SCREEN in verbose:
    #     print(f'grid={grid[:5]}...{grid[-5:]}\nquantizedVec={quantizedVec}, scale={scale}, z={z}\ndequantizedVec={dequantizedVec}')
    # if VERBOSE_DEBUG in verbose:
    #     printf(debugFile,
    #             f'grid={grid}\nvec2quantize={vec2quantize}\nquantizedVec={quantizedVec}\nscale={scale}, z={z}\ndequantizedVec={dequantizedVec}')
    #     debugFile.close()

    # todo - fix this test FP_e2 exsit??
    # # Test FP_e2 grid
    # fxpSettingStr = 'FP_e2'
    # grid = getAllValsFxp(
    #     fxpSettingStr=fxpSettingStr,
    #     cntrSize=cntrSize,
    #     verbose=[],
    #     signed=True
    # )
    # print(f'grid parameters:\n  fxpSettingsStr: {fxpSettingStr} | cntrSize: {cntrSize} | signed: True')
    # [quantizedVec, scale, z] = quantize(vec=vec2quantize, grid=grid)
    # dequantizedVec = dequantize(quantizedVec, scale, z)
    # if VERBOSE_PRINT_SCREEN in verbose:
    #     print(f'grid={grid[:5]}...{grid[-5:]}\nquantizedVec={quantizedVec}, scale={scale}, z={z}\ndequantizedVec={dequantizedVec}')
    # if VERBOSE_DEBUG in verbose:
    #     printf(debugFile,
    #             f'grid={grid}\nvec2quantize={vec2quantize}\nquantizedVec={quantizedVec}\nscale={scale}, z={z}\ndequantizedVec={dequantizedVec}')
    #     debugFile.close()


if __name__ == '__main__':
    try:
        testQuantization(verbose=[VERBOSE_PRINT_SCREEN])
        testQuantization_INT16(verbose=[VERBOSE_PRINT_SCREEN])
        # runCalcQuantRoundErr ()
        # plotGrids (zoomXlim=None, cntrSize=7, modes=['F2P_li_h2', 'F2P_si_h2', 'FP_e5', 'FP_e2', 'int'], scale=False)

    except KeyboardInterrupt:
        print('Keyboard interrupt.')
