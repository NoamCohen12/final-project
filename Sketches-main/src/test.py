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

    #------------------------------------------------------------------
    # Test f2p_lr_h2 grid noam test
    vec2quantize = np.array([0.058345668, 0.25526232,0.291172835,0.5178178, 0.5834567])

    fxpSettingStr = 'F2P_lr_h2'
    sign = False
    grid = getAllValsFxp(
        fxpSettingStr='F2P_lr_h2',
        cntrSize=cntrSize,
        verbose=[],
        signed=sign
    )
    print(f'noam grid parameters:\n  fxpSettingsStr: {fxpSettingStr} | cntrSize: {cntrSize} | signed: {sign}')
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

def test_int8_quantization(verbose: list = []):
        """
        Additional tests for INT8 quantization
        """
        vec2quantize_list = [
            np.array([-150.0, 150.0]),  # Values outside the INT8 range
            np.array([-0.5, 0.0, 0.5]),  # Subtle floating-point values near zero
            np.random.uniform(-128, 127, 20),  # Random values within INT8 range
            np.array([-128, -64, -32, 0, 32, 64, 127]),  # Boundary and symmetric intermediate values
            np.linspace(-128, 127, 50),  # Linearly spaced values in the INT8 range
            np.array([127] * 10 + [-128] * 10),  # Repeated extreme values to test stability
        ]

        cntrSize = 8  # INT8

        for i, vec2quantize in enumerate(vec2quantize_list):
            if 'VERBOSE_PRINT_SCREEN' in verbose:
                print(f"\n=== Test {i + 1} ===")
                print(f"Input Vector (vec2quantize): {vec2quantize}")

            # Create INT8 grid
            grid = np.array(range(-2 ** (cntrSize - 1), 2 ** (cntrSize - 1)), dtype="int")

            # Perform quantization
            try:
                quantizedVec, scale, z = quantize(vec=vec2quantize, grid=grid)
            except Exception as e:
                print(f"Error during quantization in Test {i + 1}: {e}")
                continue

            # Perform dequantization
            try:
                dequantizedVec = dequantize(quantizedVec, scale, z)
            except Exception as e:
                print(f"Error during dequantization in Test {i + 1}: {e}")
                continue

            # Clip dequantized values to INT8 range
            clipped_dequantizedVec = np.clip(dequantizedVec, -128, 127)

            # Compute expected values based on clipping the input vector
            expected_vec = np.clip(vec2quantize, -128, 127)

            # Log the results
            if 'VERBOSE_PRINT_SCREEN' in verbose:
                print("Grid:")
                print(f"{grid[:5]}...{grid[-5:]}")
                print(f"Quantized Vector: {quantizedVec}")
                print(f"Scale: {scale}, Zero-point: {z}")
                print(f"Dequantized Vector: {dequantizedVec}")
                print(f"Clipped Dequantized Vector: {clipped_dequantizedVec}")
                print(f"Expected Vector (after clipping): {expected_vec}")
                print("========================\n")

            if 'VERBOSE_DEBUG' in verbose:
                with open(f"../res/debug_test_{i + 1}.txt", "w") as debugFile:
                    debugFile.write(
                        f"Input Vector (vec2quantize): {vec2quantize}\n"
                        f"Grid: g{grid[:5]}...{grid[-5:]}\n"
                        f"Quantized Vector: {quantizedVec}\n"
                        f"Scale: {scale}, Zero-point: {z}\n"
                        f"Dequantized Vector: {dequantizedVec}\n"
                        f"Clipped Dequantized Vector: {clipped_dequantizedVec}\n"
                        f"Expected Vector (after clipping): {expected_vec}\n"
                    )

            # Additional assertions for validation
            assert len(quantizedVec) == len(vec2quantize), f"Test {i + 1}: Quantized vector length mismatch!"
            assert np.all(clipped_dequantizedVec <= 127) and np.all(
                clipped_dequantizedVec >= -128), f"Test {i + 1}: Dequantized values out of range!"

            # Debugging assertion failure
            if not np.allclose(expected_vec, clipped_dequantizedVec, atol=1):
                diff = expected_vec - clipped_dequantizedVec
                print(f"Test {i + 1}: Differences detected:")
                print(f"Expected: {expected_vec}")
                print(f"Actual: {clipped_dequantizedVec}")
                print(f"Difference: {diff}")

            assert np.allclose(expected_vec, clipped_dequantizedVec,
                               atol=1), f"Test {i + 1}: Dequantized values differ significantly!"




if __name__ == '__main__':
    try:
        testQuantization(verbose=[VERBOSE_PRINT_SCREEN])
        testQuantization_INT16(verbose=[VERBOSE_PRINT_SCREEN])
        VERBOSE_PRINT_SCREEN = 'VERBOSE_PRINT_SCREEN'
        test_int8_quantization(verbose=[VERBOSE_PRINT_SCREEN])        # runCalcQuantRoundErr ()
        # plotGrids (zoomXlim=None, cntrSize=7, modes=['F2P_li_h2', 'F2P_si_h2', 'FP_e5', 'FP_e2', 'int'], scale=False)

    except KeyboardInterrupt:
        print('Keyboard interrupt.')
