/**
 * Note: The returned array must be malloced, assume caller calls free().
 * C for fun
 */
int* spiralOrder(int** matrix, int matrixSize, int* matrixColSize, int* returnSize) {
    int* res = malloc(matrixSize * *matrixColSize * sizeof(int));
    int res_len = 0;
    // printf("%d\n", matrixSize);
    // printf("%d", *matrixColSize);

    int dirs[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    int dir_idx = 0;
    int curr[2] = {0, 0};
    
    while (res_len < matrixSize * *matrixColSize) {
        int i = curr[0], j = curr[1];
        res[res_len] = matrix[i][j];
        matrix[i][j] = 101;

        int di = dirs[dir_idx][0], dj = dirs[dir_idx][1];
        if (!(0 <= i + di && i + di < matrixSize && 0 <= j + dj && j + dj < *matrixColSize && matrix[i + di][j + dj] <= 100)) {
            dir_idx = (dir_idx + 1) % 4;
            di = dirs[dir_idx][0];
            dj = dirs[dir_idx][1];
        }
        curr[0] = i + di;
        curr[1] = j + dj;

        res_len++;
    }

    *returnSize = res_len;

    // for (int i = 0; i < res_len; i++) {
    //     printf("%d ", res[i]);
    // }
    return res;
}