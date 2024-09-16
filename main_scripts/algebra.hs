import Data.Matrix

main :: IO ()
main = do
    let matrix = fromLists [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    let determinant = detLU matrix
    putStrLn $ "Determinant: " ++ show determinant