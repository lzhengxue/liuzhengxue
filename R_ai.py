import numpy as np
import time

t1 = time.time()
class Reforcement_ai():
    def __init__(self, table):
        self.playerjudger = None
        self.table_2d = None  # 15*15
        self.color = None
        self.anticolor = None
        self.color_dict = {'White': 1, 'Black': -1, 'Blank': 0}
        self.t1 = None

    # 返回下子坐标 1-15
    def xiazi(self, playerjudger, color, step, table4=None):
        # 如果ai是黑子
        if color == 'Black':
            self.color = 'Black'
            self.anticolor = 'White'
        else:  # 如果ai是白子
            self.color = 'White'
            self.anticolor = 'Black'

        # 根据奖励分数来选择下子点

        self.playerjudger = playerjudger  # 用于禁手判断
            # 注意：此时table_ed是个16*16的数组，0-14是有效区域
            # 将数组转移到索引为0-14，更易于操作
        self.table_2d = playerjudger.table_2d[1:16, 1:16]  # 把ai下子前的棋盘上的情况告诉table_2d
        self.t1 = time.time()  # 计时

        stl4 = np.array(self.tablelevel(self.table_2d, self.color_dict[self.color]))
        atl4 = np.array(self.tablelevel(self.table_2d, self.color_dict[self.anticolor]))
        #print(self.table_2d)

        #print(stl4)
        #print(atl4)

        sx, sy, sz = self.VCF(self.table_2d, stl4, atl4, self.color_dict[self.color], 5)
        print([sx, sy, sz])
        if sx == 1:
            print('vcf')
            return sy + 1, sz + 1
        ax, ay, az = self.VCF(self.table_2d, atl4, stl4, self.color_dict[self.anticolor], 5)
        print([ax, ay, az])
        if ax == 1:
            print('对方vcf')
            return ay + 1, az + 1

        px, py = self.try3(self.table_2d, stl4, atl4, self.color_dict[self.color])
        print([px, py])
        if px > 15:
            print('vct')
            return px - 14, py + 1

        fx, fy = self.tryf3(self.table_2d, stl4, atl4, self.color_dict[self.color])
        print([fx, fy])
        if fx > 0:
            print('做vcf')
            return fx + 1, fy + 1

        if px > 0:
            print('假装vct')
            return px + 1, py + 1

        kx, ky = self.try3(self.table_2d, atl4, stl4, self.color_dict[self.anticolor])
        print([kx, ky])
        if kx > 15:
            print('对方vct')
            return kx - 14, ky + 1



        a = np.array(self.covalue_table(self.table_2d, self.color_dict[self.color]))  # 计算我方的分值
        b = np.array(self.covalue_table(self.table_2d, self.color_dict[self.anticolor])) # 计算敌方的分值
        c = np.array([a, b])
        pos = np.unravel_index(np.argmax(c), c.shape)  # 计算最大值所在的坐标

        print('算值')

        # 调整落子坐标
        pos_x, pos_y = pos[1] + 1, pos[2] + 1
        print(c)
        return pos_x, pos_y

    def uop_b(self, a_value, table_value):  # 黑棋单位运算unit operation
        if table_value == -1:
            return a_value
        else:
            return 1

    def uop_w(self, a_value, table_value):  # 白棋单位运算unit operation
        if table_value == 1:
            return a_value
        else:
            return 1

    def covalue(self, table_list, color):  # 对一个给定的list做卷积，将棋盘划分为一条一条一维list后做卷积
        kernel = [1, 20, 15, 10, 5] # 卷积核
        kernelanti = [1, 4, 3, 2, 1.25]
        value_blank = []
        if color == -1:  # 黑棋的情况
            new_list = np.pad(table_list, (4, 4), 'constant', constant_values=(1, 1))  # 将边界设为白旗
            for i in range(len(table_list)):
                if table_list[i] == 0:
                    a = 1
                    for j in range(1, 5):  # 分别从左右两个方向将kernel作用到list上
                        if new_list[i + 4 - j] == 1:# 遇到白旗就中断计算
                            if new_list[i+5-j] == -1 or j == 1:
                                a = a/kernelanti[j]
                            break
                        else:
                            a *= self.uop_b(kernel[j], new_list[i + 4 - j])
                    for j in range(1, 5):
                        if new_list[i + 4 + j] == 1:
                            if new_list[i + 3 + j] == -1 or j == 1:
                                a = a/kernelanti[j]
                            break
                        else:
                            a *= self.uop_b(kernel[j], new_list[i + 4 + j])

                    value_blank.append(round(a))
                else:
                    value_blank.append(table_list[i])
        else:  # 白旗的情况
            new_list = np.pad(table_list, (4, 4), 'constant', constant_values=(-1, -1))  # 将边界设为黑棋
            for i in range(len(table_list)):
                if table_list[i] == 0:
                    a = 1
                    for j in range(1, 5):
                        if new_list[i + 4 - j] == -1:  # 遇到黑旗就中断计算
                            if new_list[i + 5 - j] == 1 or j == 1:
                                a = a / kernelanti[j]
                            break
                        else:
                            a *= self.uop_w(kernel[j], new_list[i + 4 - j])
                    for j in range(1, 5):
                        if new_list[i + 4 + j] == -1:
                            if new_list[i + 3 + j] == 1 or j == 1:
                                a = a / kernelanti[j]
                            break
                        else:
                            a *= self.uop_w(kernel[j], new_list[i + 4 + j])
                    value_blank.append(round(a))
                else:
                    value_blank.append(table_list[i])
        return value_blank

    def covalue_table(self, table_2d, color):
        value = [[0] * 15 for i in range(15)]
        value_h = []
        value_v = [[0] * 15 for i in range(15)]
        value_d = [[0] * 15 for i in range(15)]
        value_m = [[0] * 15 for i in range(15)]

        # 从四个方向分别对棋盘计算每个点的单色子力密度，horizontal, vertical, major diagonal, and minor diagonal
        # horizontal 横向
        for i in range(15):
            value_h.append(self.covalue(table_2d[i], color))
        print(np.array(value_h))

        # vertical 纵向
        for i in range(15):
            p = []
            r = []
            for j in range(15):
                p.append(table_2d[j][i])
            r = self.covalue(p, color)
            for j in range(15):
                value_v[j][i] = r[j]
        print(np.array(value_v))

        # major diagonal 主对角线方向
        for i in range(21):
            p = []
            if i < 11:
                for j in range(i + 5):
                    p.append(table_2d[j][10 - i + j])
                r = self.covalue(p, color)
                for j in range(i + 5):
                    value_d[j][10 - i + j] = r[j]
            else:
                for j in range(25 - i):
                    p.append(table_2d[i - 10 + j][j])
                r = self.covalue(p, color)
                for j in range(25 - i):
                    value_d[i - 10 + j][j] = r[j]
        print(np.array(value_d))
        # minor diagonal 次对角线方向
        for i in range(21):
            p = []
            if i < 11:
                for j in range(i + 5):
                    p.append(table_2d[4 + i - j][j])
                r = self.covalue(p, color)
                for j in range(i + 5):
                    value_m[4 + i - j][j] = r[j]
            else:
                for j in range(25 - i):
                    p.append(table_2d[14 - j][i - 10 + j])
                r = self.covalue(p, color)
                for j in range(25 - i):
                    value_m[14 - j][i - 10 + j] = r[j]
        print(np.array(value_m))

        # 将四个方向的子力密集程度的值相乘，得到最终的子力密集程度值
        for i in range(15):
            for j in range(15):
                if value_d[i][j] == 0:
                    value_d[i][j] = 1
                if value_m[i][j] == 0:
                    value_m[i][j] = 1
                if table_2d[i][j] == 0:
                    value[i][j] = self.max2(value_h[i][j], value_v[i][j], value_d[i][j], value_m[i][j])

                else:
                    value[i][j] = table_2d[i][j]
                # 禁手调整
                if table_2d[i, j] == 0:
                    if color == -1:
                        self.table_2d[i][j] = self.color_dict['Black']  # 一定要先下子
                        if self.playerjudger.check_forbidden([j + 1, i + 1], 'Black', False) == False:
                            value[i][j] = 0
                        elif self.playerjudger.check_win([j + 1, i + 1], 'Black', False) == True:
                            value[i][j] += 100000000
                        self.table_2d[i][j] = self.color_dict['Blank']
                    else:
                        self.table_2d[i][j] = self.color_dict['White']  # 一定要先下子
                        if self.playerjudger.check_win([j + 1, i + 1], 'White', False) == True:
                            value[i][j] += 100000000
                        self.table_2d[i][j] = self.color_dict['Black']  # 一定要先下子
                        if self.playerjudger.check_forbidden([j + 1, i + 1], 'Black', False) == False:
                            value[i][j] = 0
                        self.table_2d[i][j] = self.color_dict['Blank']

        return value



    def max2(self, a, b, c, d):
        L = [a, b, c, d]
        L.sort()
        return L[2] * L[3]

    def A5(self, list, index, color):
        if sum(list[index - 4: index]) == 4 * color and list[index - 5] == 0:
            return 10
        if sum(list[index + 1: index + 5]) == 4 * color and list[index + 5] == 0:
            return 10
        for i in range(5):
            if sum(list[(index - 4 + i):(index + i + 1)]) == 4 * color:
                return 1
        return 0

    def A4(self, list4, index, color):
        list4[index] = color
        a = 0
        for j in range(-4, 5):
            if list4[index + j] == 0:
                a += self.A5(list4, index + j, color)
                if a > 9:
                    list4[index] = 0
                    return 10
            else:
                continue
        if a > 0:
            list4[index] = 0
            return 1
        list4[index] = 0
        return 0

    def A3(self, list3, index, color):
        list3[index] = color
        a = 0
        for j in range(-2, 3):
            if list3[index + j] == 0:
                a += self.A4(list3, index + j, color)
                if a > 9:
                    list3[index] = 0
                    return 10
            else:
                continue
        if a > 0:
            list3[index] = 0
            return 1
        list3[index] = 0
        return 0

    def A2(self, list2, index, color):
        list2[index] = color
        a = 0
        if list2[index + 1] == 0:
            a = self.A3(list2, index + 1, color)
            if a > 9:
                list2[index] = 0
                return 10
        if list2[index - 1] == 0:
            a += self.A3(list2, index - 1, color)
            if a > 9:
                list2[index] = 0
                return 10
        if a > 0:
            list2[index] = 0
            return 1
        list2[index] = 0
        return 0

    def level(self, list, index, color):
        a = 0

        a = self.A5(list, index, color)
        if a > 0:
            return 8
        a = self.A4(list, index, color)
        if a == 10:
            return 7
        elif a == 1:
            return 6
        a = self.A3(list, index, color)
        if a == 10:
            return 5
        elif a == 1:
            return 4
        a = self.A2(list, index, color)
        if a == 10:
            return 3
        elif a == 1:
            return 2
        return 2

    def listlevel(self, table_list, color):

        new_list = np.pad(table_list, (4, 4), 'constant', constant_values=(-color, -color))
        value_blank = []
        for i in range(len(table_list)):
            if table_list[i] == -1:
                value_blank.append(-1)
                continue

            elif table_list[i] == 1:
                value_blank.append(-2)
                continue
            elif table_list[i] == 0:
                b, c = 0, [0, 10] # b代表本色子的数量，c代表这一行中本色棋子的生存长度，如果长度小于5，则认为这一位置没有任何优先级
                for j in range(1, 5):
                    if new_list[i + 4 - j] == -color:
                        c[0] = 5 - j
                        break # 遇到对方子就停止计算
                    elif new_list[i + 4 - j] == color:
                        b += 1
                for j in range(1, 5):
                    if new_list[i + 4 + j] == -color:
                        c[1] = 5 + j
                        break # 遇到对方子就停止计算
                    elif new_list[i + 4 + j] == color:
                        b += 1
                if c[1] - c[0] < 6: # 如果长度小于5，则认为这一位置没有任何优先级
                    value_blank.append(0)
                    continue
                if b == 0: # 这一方向上没有同色子，则认为这一位置的优先级为一
                    value_blank.append(1)
                    continue
                if b == 1: # 这一方向仅有一个同色子，则认为这一位置的优先级为二（2代表眠二）
                    value_blank.append(2)
                    continue
                else: # 其他情况下，调用level函数确定该位置的优先级
                    value_blank.append(self.level(new_list, i + 4, color))
        # print(new_list)
        return value_blank

    def tablelevel(self, table_2d, color):
        # value = [[0]*15 for i in range(15)]
        value = []
        value_h = []
        value_v = [[0] * 15 for i in range(15)]
        value_d = [[0] * 15 for i in range(15)]
        value_m = [[0] * 15 for i in range(15)]

        # 从四个方向分别对棋盘计算每个点的单色子力密度，horizontal, vertical, major diagonal, and minor diagonal
        # horizontal 横向
        for i in range(15):
            value_h.append(self.listlevel(table_2d[i], color))
        # print(np.array(value_h))
        value.append(value_h)
        # vertical 纵向

        for i in range(15):
            p = []
            r = []
            for j in range(15):
                p.append(table_2d[j][i])
            r = self.listlevel(p, color)
            for j in range(15):
                value_v[j][i] = r[j]
        # print(np.array(value_v))
        value.append(value_v)
        # major diagonal 主对角线方向
        for i in range(21):
            p = []
            if i < 11:
                for j in range(i + 5):
                    p.append(table_2d[j][10 - i + j])
                r = self.listlevel(p, color)
                for j in range(i + 5):
                    value_d[j][10 - i + j] = r[j]
            else:
                for j in range(25 - i):
                    p.append(table_2d[i - 10 + j][j])
                r = self.listlevel(p, color)
                for j in range(25 - i):
                    value_d[i - 10 + j][j] = r[j]
        # print(np.array(value_d))
        value.append(value_d)

        # minor diagonal 次对角线方向
        for i in range(21):
            p = []
            if i < 11:
                for j in range(i + 5):
                    p.append(table_2d[4 + i - j][j])
                r = self.listlevel(p, color)
                for j in range(i + 5):
                    value_m[4 + i - j][j] = r[j]
            else:
                for j in range(25 - i):
                    p.append(table_2d[14 - j][i - 10 + j])
                r = self.listlevel(p, color)
                for j in range(25 - i):
                    value_m[14 - j][i - 10 + j] = r[j]
        value.append(value_m)
        return value

    def fast8(self, tl4):
        for i in range(4):
            for j in range(15):
                for k in range(15):
                    if tl4[i][j][k] == 8:
                        return [j, k]
        return []

    def find8(self, tl4):
        VCF_points = []
        for i in range(4):
            for j in range(15):
                for k in range(15):
                    if tl4[i][j][k] == 8:
                        VCF_points.append([8, j, k, self.max3(tl4[0][j][k], tl4[1][j][k], tl4[2][j][k], tl4[3][j][k])])
                    elif tl4[i][j][k] == 7:
                        VCF_points.append([7, j, k, self.max3(tl4[0][j][k], tl4[1][j][k], tl4[2][j][k], tl4[3][j][k])])
                    elif tl4[i][j][k] == 6:
                        VCF_points.append([6, j, k, self.max3(tl4[0][j][k], tl4[1][j][k], tl4[2][j][k], tl4[3][j][k])])
        return sorted(VCF_points, key=lambda x: (x[0], x[3]), reverse=True)

    def find5(self, tl4):
        VCF_points = []
        for i in range(4):
            for j in range(15):
                for k in range(15):
                    if tl4[i][j][k] == 5:
                        VCF_points.append([5, j, k, self.max3(tl4[0][j][k], tl4[1][j][k], tl4[2][j][k], tl4[3][j][k])])
        return sorted(VCF_points, key=lambda x: (x[3]), reverse=True)

    def find4(self, tl4):
        VCF_points = []
        for i in range(4):
            for j in range(15):
                for k in range(15):
                    if tl4[i][j][k] == 4:
                        VCF_points.append([4, j, k, self.max3(tl4[0][j][k], tl4[1][j][k], tl4[2][j][k], tl4[3][j][k])])
        return sorted(VCF_points, key=lambda x: (x[3]), reverse=False)

    def pointrelevel(self, table_2d, stl4, atl4, c, color):
        table_2d[c[0]][c[1]] = color

        stl4[0][c[0]] = self.listlevel(table_2d[c[0]], color)
        atl4[0][c[0]] = self.listlevel(table_2d[c[0]], -color)

        v = []

        for j in range(15):
            v.append(table_2d[j][c[1]])
        p = self.listlevel(v, color)
        for i in range(15):
            stl4[1][i][c[1]] = p[i]
        p = self.listlevel(v, -color)
        for i in range(15):
            atl4[1][i][c[1]] = p[i]

        i = c[0] - c[1] + 10
        p = []
        if i < 11:
            for j in range(i + 5):
                p.append(table_2d[j][10 - i + j])
            tem1 = self.listlevel(p, color)
            for j in range(i + 5):
                stl4[2][j][10 - i + j] = tem1[j]
            tem2 = self.listlevel(p, -color)
            for j in range(i + 5):
                atl4[2][j][10 - i + j] = tem2[j]
        else:
            for j in range(25 - i):
                p.append(table_2d[i - 10 + j][j])
            tem3 = self.listlevel(p, color)
            for j in range(25 - i):
                stl4[2][i - 10 + j][j] = tem3[j]
            tem4 = self.listlevel(p, -color)
            for j in range(25 - i):
                atl4[2][i - 10 + j][j] = tem4[j]

        i = c[0] + c[1] - 4
        p = []
        if i < 11:
            for j in range(i + 5):
                p.append(table_2d[4 + i - j][j])
            r = self.listlevel(p, color)
            for j in range(i + 5):
                stl4[3][4 + i - j][j] = r[j]
            r = self.listlevel(p, -color)
            for j in range(i + 5):
                atl4[3][4 + i - j][j] = r[j]
        else:
            for j in range(25 - i):
                p.append(table_2d[14 - j][i - 10 + j])
            r = self.listlevel(p, color)
            for j in range(25 - i):
                stl4[3][14 - j][i - 10 + j] = r[j]
            r = self.listlevel(p, -color)
            for j in range(25 - i):
                atl4[3][14 - j][i - 10 + j] = r[j]

    def VCF(self, table_2d, stl4, atl4, color, depth):
        if depth == 0:
            return 0, -1, -1 #深度结束
        c = self.find8(stl4)
        if len(c) == 0:
            return 0, -2, -2 #没有可冲四点
        if c[0][0] == 8:
            return 1, c[0][1], c[0][2] #已经可以连五胜
        d = self.fast8(atl4)
        # print(d)
        if len(d) != 0:
            return 0, d[0], d[1] #对方有四，需挡
        if c[0][0] == 7:
            return 1, c[0][1], c[0][2] #活四取胜

        for j in range(min(5, len(c))):
            temptable = np.array([[table_2d[j][k] for k in range(15)] for j in range(15)])
            tempstl4 = np.array([[[stl4[i][j][k] for k in range(15)] for j in range(15)] for i in range(4)])
            tempatl4 = np.array([[[atl4[i][j][k] for k in range(15)] for j in range(15)] for i in range(4)])
            self.pointrelevel(temptable, tempstl4, tempatl4, [c[j][1], c[j][2]], color)
            self.pointrelevel(temptable, tempatl4, tempstl4, self.fast8(tempstl4), -color)
            x, y, z = self.VCF(temptable, tempstl4, tempatl4, color, depth - 1)
            if x == 1:
                return 1, c[j][1], c[j][2] #有连续冲四取胜

        return 0, -3, -3 #出现在最外层代表深度耗尽

    def max3(self, a, b, c, d):
        L = [a, b, c, d]
        L.sort()
        return L[3] * 1000 + L[2] * 100 + L[1] * 10 + L[0]

    def try3(self, table_2d, stl4, atl4, color):
        c = self.find5(stl4)
        if len(c) == 0:
            return -1, -1
        p = [[] for a in range(len(c))]
        for r in range(min(8, len(c))):
            temptable = np.array([[table_2d[j][k] for k in range(15)] for j in range(15)])
            tempstl4 = np.array([[[stl4[i][j][k] for k in range(15)] for j in range(15)] for i in range(4)])
            tempatl4 = np.array([[[atl4[i][j][k] for k in range(15)] for j in range(15)] for i in range(4)])
            self.pointrelevel(temptable, tempstl4, tempatl4, [c[r][1], c[r][2]], color)
            #print(temptable)
            d = self.find8(tempstl4)

            for s in range(min(8, len(d))):
                if d[s][0] == 6:
                    break
                temptable0 = np.array([[temptable[j][k] for k in range(15)] for j in range(15)])
                tempstl40 = np.array([[[tempstl4[i][j][k] for k in range(15)] for j in range(15)] for i in range(4)])
                tempatl40 = np.array([[[tempatl4[i][j][k] for k in range(15)] for j in range(15)] for i in range(4)])
                self.pointrelevel(temptable0, tempatl40, tempstl40, [d[s][1], d[s][2]], -color)
                #print(temptable0)
                x, y, z = self.VCF(temptable0, tempstl40, tempatl40, color, 4)
                if x == 1:
                    p[r].append(y)
                    p[r].append(z)
            #print(d)
        #print(p)
        for b in range(len(p)):
            if len(p[b]) == 4:
                return 15 + c[b][1], c[b][2]
        for b in range(len(p)):
            if len(p[b]) == 2:
                return c[b][1], c[b][2]
        return -1, -1

    def tryf3(self, table_2d, stl4, atl4, color):
        c = self.find4(stl4)
        if len(c) == 0:
            return -1, -1
        for r in range(min(8, len(c))):
            temptable = np.array([[table_2d[j][k] for k in range(15)] for j in range(15)])
            tempstl4 = np.array([[[stl4[i][j][k] for k in range(15)] for j in range(15)] for i in range(4)])
            tempatl4 = np.array([[[atl4[i][j][k] for k in range(15)] for j in range(15)] for i in range(4)])
            self.pointrelevel(temptable, tempstl4, tempatl4, [c[r][1], c[r][2]], color)
            x, y, z = self.VCF(temptable, tempstl4, tempatl4, color, 4)
            if x == 1:
                return c[r][1], c[r][2]
        return -1, -1
