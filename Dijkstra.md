## 什么是dijkstra算法？它可以用来解决什么问题？
Dijkstra算法是一种用于求解**带权有向图**中**单源**最短路径的经典算法。通俗点说，就是通过这个算法可以求出**从源节点**到**其他各个节点**的最短距离。是的，没说错，就是其他各个节点。

![dij](https://github.com/TanLian/algorithm/blob/main/img/dij2.png)

假设有个图如上图所示，通过dijkstra算法就可以轻松求出从任意一个节点开始到其他各个节点的最短距离。

有以下几点需要注意：

1. 单源：源节点只有一个。也就是说如果你既要求出从节点0到节点2的最短距离，也要求出从节点4到节点5的最短距离，那么这个时候你就要执行两遍dijkstra算法
2. 带权重：可以理解成两个节点之间的距离，我们最终要求的就是从源节点到其它节点的最短距离。**需要注意的是这个权重不能为负数**
3. 有向：指的是边是单向的。如上图中节点0到节点1是可达的且权重为9，反之则不成立

## dijkstra算法的限制
权重不能为负数

## dijkstra算法的过程
1. 创建两个集合S和U。S代表已知最短路径的节点集合，初始为空；U代表未知最短路径的节点集合，初始包含所有节点。
2. 将起始节点加入集合S中，并设置该节点到自身的距离为0。
3. 对于除起始节点外的每个节点，初始化其距离为无穷大（或-1）。
4. 从集合U中找到一个距离起始节点最近的节点v，并将其加入集合S中。
5. 对于节点v的每个相邻节点w，如果通过v可以获得更短的距离，则更新w的距离值，并将v作为w的前驱节点。
6. 重复步骤4和5，直到集合U为空或者所有节点的距离都已确定为止。
7. 根据每个节点的前驱节点，可以回溯出从起始节点到该节点的最短路径。

当然第2步和第3步是可以对调的。

## dijkstra算法的实现
我们先定义函数的签名为`func dijkstra(n, start int, edges []Edge) []int`，其中`n`代表节点总数，如示例中的图因为有7个节点，所以n为7。`start`表示源节点，如示例中是要求出节点0到其他节点的最短距离，所以`start`为0。`edges`表示边的集合，其中`Edge`是个结构体，定义如下：

```go
// Edge 代表一条有向边
type Edge struct {
	from   int // 起点
	to     int // 终点
	weight int // 权重
}
```

返回值是一个int类型的数组，数组长度为`n`，表示从源节点到其他各个节点的最短距离。

定义好了函数签名后，我们看看如何实现这个函数。

虽然算法中说需要创建两个集合S和U，但是我们不必真的创建这两个集合。

我们只需创建一个长度为`n`的int类型的数组`ret`，表示从源节点到其他各个节点的最短距离，这也是整个函数的返回值。**`ret[i]`为-1代表节点i在U集合中，否则就在S集合中**。

初始时`ret`的每一项都为-1，表示每个节点都不可达（对应步骤3）；

然后将下标为`start`的`ret`设置为0：`ret[start] = 0`，代表起始节点到起始节点的距离为0（对应步骤2）。

然后就是最关键的步骤4和步骤5了。我们想象一下，对于上图，节点0有3条相邻边，分别是到节点1（距离为9），到节点4（距离为4），到节点3（距离为8）。如果你只能看到这4个节点看不到后面节点的话，你会选择走哪条路？肯定会选择节点4，因为就目前来说，这条路的路径最短，这也体现了[贪心法](https://github.com/TanLian/algorithm/blob/main/%E8%B4%AA%E5%BF%83%E6%B3%95.md)的思想（局部最优得到全局最优）。这一步可以通过**[小顶堆/优先队列](https://github.com/TanLian/algorithm/blob/main/%E5%A0%86%EF%BC%88%E4%BC%98%E5%85%88%E9%98%9F%E5%88%97%EF%BC%89%20.md)**来做，当然你也可以通过一个for循环来一个个遍历得到最小的相邻边，但是这样的话时间复杂度会更高些。

得到节点4之后，我们将节点4加入到S集合（即设置`ret[4] = 4`），代表从节点0到节点4的最短距离为4。然后我们再以节点4为驱动点，找到它的所有相邻边，看看通过节点4能不能缩小从起始点到它的相邻节点（对应此图为节点5）的最短距离。

依次循环。

golang代码如下:
```go
const unReachable = -1 // 不可达

// Edge 代表一条有向边
type Edge struct {
	from   int // 起点
	to     int // 终点
	weight int // 权重
}

// dijkstra dijkstra算法的go实现
// n: 节点总数
// start: 起始节点
// edges: 边
// 返回值：从start到各个节点的最短距离，如果节点不可达则为-1
func dijkstra(n, start int, edges []Edge) []int {
	ret := make([]int, n) // 返回值

	// 各个节点都初始化为不可达
	for i := 0; i < n; i++ {
		ret[i] = unReachable
	}

	// 起点到起点的最短距离为0
	ret[start] = 0

	// 构建相邻边
	neighbours := makeNeighbours(n, edges)

	// 创建小顶堆
	var hp MinHeap
	heap.Init(&hp)
	heap.Push(&hp, [2]int{start, 0}) // 下标0表示节点id，下标1表示从起点到该节点的最短距离

	for hp.Len() > 0 {
		item := heap.Pop(&hp).([2]int)
		var (
			id  = item[0] // 当前节点id
			dis = item[1] // 从起点到当前节点的最短距离
		)

		// 如果之前有更小的路径到达该节点，则跳过
		if ret[id] != unReachable && dis > ret[id] {
			continue
		}

		// 走到这说明要么是第一次到达该节点，要么当前这条路径是达到该点的最短距离
		ret[id] = dis

		// 将该节点的相邻节点都加入到小顶堆中
		for _, v := range neighbours[id] {
			// 如果相邻节点之前未到达过 或者 当前这条路径比之前的任何路径的距离都要短
			if ret[v.to] == unReachable || ret[id]+v.weight < ret[v.to] { // 加上这个条件实际上是过滤掉了那些永远不可能成为最短距离的路径，达到了剪枝的效果
				heap.Push(&hp, [2]int{v.to, ret[id] + v.weight})
			}
		}
	}
	return ret
}

// makeNeighbours 构建相邻边
func makeNeighbours(n int, edges []Edge) [][]Edge {
	neighbours := make([][]Edge, n)
	for _, v := range edges {
		neighbours[v.from] = append(neighbours[v.from], v)
	}
	return neighbours
}

// MinHeap 小顶堆
type MinHeap [][2]int

func (pq MinHeap) Len() int { return len(pq) }

func (pq MinHeap) Less(i, j int) bool {
	return pq[i][1] < pq[j][1]
}

func (pq MinHeap) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}

func (pq *MinHeap) Push(x interface{}) {
	e := x.([2]int)
	*pq = append(*pq, e)
}

func (pq *MinHeap) Pop() interface{} {
	old := *pq
	n := len(old)
	node := old[n-1]
	*pq = old[:n-1]
	return node
}
```

rust代码如下：
```rust
const UNREACHABLE: i32 = -1; // 不可达

#[derive(Debug)]
struct Edge {
    from: usize, // 起点
    to: usize,   // 终点
    weight: i32, // 权重
}

// dijkstra dijkstra算法的rust实现
// n: 节点总数
// start: 起始节点
// edges: 边
// 返回值：从start到各个节点的最短距离，如果节点不可达则为-1
fn dijkstra(n: usize, start: usize, edges: &[Edge]) -> Vec<i32> {
    let mut ret = vec![UNREACHABLE; n]; // 返回值

    // 起点到起点的最短距离为0
    ret[start] = 0;

    // 构建相邻边
    let neighbours = make_neighbours(n, edges);

    // 创建小顶堆
    let mut hp = std::collections::BinaryHeap::new();

    hp.push(std::cmp::Reverse((0, start))); // 下标0表示从起点到该节点的最短距离，下标1表示节点id

    while let Some(top) = hp.pop() {
        let std::cmp::Reverse((dis, id)) = top;

        // 如果之前有更小的路径到达该节点，则跳过
        if ret[id] != UNREACHABLE && dis > ret[id] {
            continue;
        }

        // 走到这说明要么是第一次到达该节点，要么当前这条路径是达到该点的最短距离
        ret[id] = dis;

        // 将该节点的相邻节点都加入到小顶堆中
        for v in &neighbours[id] {
            // 如果相邻节点之前未到达过 或者 当前这条路径比之前的任何路径的距离都要短
            if ret[v.to] == UNREACHABLE || ret[id] + v.weight < ret[v.to] { // 加上这个条件实际上是过滤掉了那些永远不可能成为最短距离的路径，达到了剪枝的效果
                hp.push(std::cmp::Reverse((ret[id] + v.weight, v.to)));
            }
        }
    }
    return ret;
}

// make_neighbours 构建相邻边
fn make_neighbours(n: usize, edges: &[Edge]) -> Vec<Vec<&Edge>> {
    let mut neighbours = vec![vec![]; n];
    for v in edges {
        neighbours[v.from].push(v.clone());
    }
    return neighbours;
}
```

## [网络延迟时间](https://leetcode.cn/problems/network-delay-time)

### 解法1：dijkstra算法
这个题就是典型的dijkstra算法，我们只需要求出从节点`k`到其它各个节点的最短路径，然后找出最大值即可。边界情况处理：如果有任何一个节点不可达，则返回-1.

golang代码如下：
```go
func networkDelayTime(times [][]int, n int, k int) int {
	var edges []Edge
	for _, v := range times {
		edges = append(edges, Edge{v[0]-1, v[1]-1, v[2]})
	}
	distTo := dijkstra(n, k-1, edges)
	
	var ret int
	for _, v := range distTo {
		if v == unReachable { // 如果有不可达的节点，则返回-1
			return unReachable
		}
		if v > ret {
			ret = v
		}
	}
	return ret
}
```

## [最小体力消耗路径](https://leetcode.cn/problems/path-with-minimum-effort/)
### 解法1：dijkstra算法
我们可以把二维数组的**每个元素看成一个点**，比如将第0行第0列元素看成节点0，将第0行第1列元素看成节点1，依次类推，**两个点之间的差值的绝对值看成是这两个点形成的边（edge）的权重（weight）**，然后需要求出的是起始点到终点的权重，这样就可以将这个问题转换成可以用dijkstra算法解决的问题。

注意题目中说的是**高度差绝对值**，正是有了这个条件我们才能用dijkstra算法来解决此题。

![dij3](https://github.com/TanLian/algorithm/blob/main/img/dij3.png)

有人可能会说dijkstra算法要求**边是有向的**，而你这个图看起来边是没有方向的。其实我这个图边也是有方向的，只是是**双向**的，如从节点0到节点1的权重为1，从节点1到节点0的权重也为1。因为是双向的，所以我就没把方向画出来。

不过这个跟dijkstra算法有一些许不同，之前是这样的，比如说从节点0到节点1的权重是1，节点1到节点2的权重是2，则节点0到节点2的权重则为`1+2=3`，也就是说**之前是累加的关系**，所以之前的代码我们是这么写的：
```go
// 如果相邻节点之前未到达过 或者 当前这条路径比之前的任何路径的距离都要短
if ret[v.to] == unReachable || ret[id]+v.weight < ret[v.to] { // 加上这个条件实际上是过滤掉了那些永远不可能成为最短距离的路径，达到了剪枝的效果
	heap.Push(&hp, [2]int{v.to, ret[id] + v.weight})
}
```

而本题规定，**一条路径耗费的 体力值 是路径上相邻格子之间 高度差绝对值 的 最大值 决定的。**，也就是说本题不是累加的关系了，而是**取路径上边的权重的最大值**。如节点0到节点1的权重是1，节点1到节点2的权重是2，那么节点0到节点2的权重也是2，所以我们需要把代码改成下面这样：
```go
w := max(ret[id], v.weight) // 这里取整个路径上的边的最大权重
if ret[v.to] == unReachable || w < ret[v.to] { 
	heap.Push(&hp, [2]int{v.to, w})
}
```
除了这点之外，其它的代码就完全一样了。

完整代码如下：

golang版：
```go
func minimumEffortPath(heights [][]int) int {
	rows := len(heights)
	cols := len(heights[0])
	var edges []Edge
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if i > 0 {
				edges = append(edges, Edge{from: i*cols + j, to: (i-1)*cols + j, weight: abs(heights[i][j] - heights[i-1][j])})
			}
			if i+1 < rows {
				edges = append(edges, Edge{from: i*cols + j, to: (i+1)*cols + j, weight: abs(heights[i][j] - heights[i+1][j])})
			}
			if j > 0 {
				edges = append(edges, Edge{from: i*cols + j, to: i*cols + j - 1, weight: abs(heights[i][j] - heights[i][j-1])})
			}
			if j+1 < cols {
				edges = append(edges, Edge{from: i*cols + j, to: i*cols + j + 1, weight: abs(heights[i][j] - heights[i][j+1])})
			}
		}
	}

	dij := dijkstra(rows*cols, 0, edges)
	return dij[len(dij)-1]
}

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

const unReachable = -1 // 不可达

// Edge 代表一条有向边
type Edge struct {
	from   int // 起点
	to     int // 终点
	weight int // 权重
}

// dijkstra dijkstra算法的go实现
// n: 节点总数
// start: 起始节点
// edges: 边
// 返回值：从start到各个节点的最短距离，如果节点不可达则为-1
func dijkstra(n, start int, edges []Edge) []int {
	ret := make([]int, n) // 返回值

	// 各个节点都初始化为不可达
	for i := 0; i < n; i++ {
		ret[i] = unReachable
	}

	// 起点到起点的最短距离为0
	ret[start] = 0

	// 构建相邻边
	neighbours := makeNeighbours(n, edges)

	// 创建小顶堆
	var hp MinHeap
	heap.Init(&hp)
	heap.Push(&hp, [2]int{start, 0}) // 下标0表示节点id，下标1表示从起点到该节点的最短距离

	for hp.Len() > 0 {
		item := heap.Pop(&hp).([2]int)
		var (
			id  = item[0] // 当前节点id
			dis = item[1] // 从起点到当前节点的最短距离
		)

		// 如果之前有更小的路径到达该节点，则跳过
		if ret[id] != unReachable && dis > ret[id] {
			continue
		}

		// 走到这说明要么是第一次到达该节点，要么当前这条路径是达到该点的最短距离
		ret[id] = dis

		// 将该节点的相邻节点都加入到小顶堆中
		for _, v := range neighbours[id] {
			// 如果相邻节点之前未到达过 或者 当前这条路径比之前的任何路径的距离都要短
			w := max(ret[id], v.weight)
			if ret[v.to] == unReachable || w < ret[v.to] { // 加上这个条件实际上是过滤掉了那些永远不可能成为最短距离的路径，达到了剪枝的效果
				heap.Push(&hp, [2]int{v.to, w})
			}
		}
	}
	return ret
}

// makeNeighbours 构建相邻边
func makeNeighbours(n int, edges []Edge) [][]Edge {
	neighbours := make([][]Edge, n)
	for _, v := range edges {
		neighbours[v.from] = append(neighbours[v.from], v)
	}
	return neighbours
}

// MinHeap 小顶堆
type MinHeap [][2]int

func (pq MinHeap) Len() int { return len(pq) }

func (pq MinHeap) Less(i, j int) bool {
	return pq[i][1] < pq[j][1]
}

func (pq MinHeap) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}

func (pq *MinHeap) Push(x interface{}) {
	e := x.([2]int)
	*pq = append(*pq, e)
}

func (pq *MinHeap) Pop() interface{} {
	old := *pq
	n := len(old)
	node := old[n-1]
	*pq = old[:n-1]
	return node
}
```

## [概率最大的路径](https://leetcode.cn/problems/path-with-maximum-probability)
### 解法1：dijkstra算法
这个可以用dijkstra算法解决，不过和之前的有两点不同：

1. 之前都是求从起点到其他点的**最短**路径，而本题是求**最大**路径，所以本题我们不能用小顶堆而应该**用大顶堆**。
2. 之前权重都是累加，比如节点1到节点2权重为1，节点2到节点3权重为3，则节点1到节点3权重为4。而本题是**累乘**，如节点1到节点2权重为0.5，节点2到节点3权重为0.5，则节点1到节点3的权重为0.25。

主要是这两点不同，当然还有其他一些小的点，如之前的权重是整形，而**本题的权重是浮点型**。

完整代码如下：

golang版：
```go
func maxProbability(n int, edges [][]int, succProb []float64, start int, end int) float64 {
	var es []Edge
	for i := 0; i < len(edges); i++ {
		es = append(es, Edge{from: edges[i][0], to: edges[i][1], weight: succProb[i]})
		es = append(es, Edge{from: edges[i][1], to: edges[i][0], weight: succProb[i]})
	}

	ret := dijkstra(n, start, es)
	if ret[end] == unReachable {
		return 0
	}
	return ret[end]
}

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

const unReachable = -1 // 不可达

// Edge 代表一条有向边
type Edge struct {
	from   int     // 起点
	to     int     // 终点
	weight float64 // 权重
}

type Item struct {
	id     int     // 节点id
	weight float64 // 从起点到该节点的权重
}

// dijkstra dijkstra算法的go实现
// n: 节点总数
// start: 起始节点
// edges: 边
// 返回值：从start到各个节点的最短距离，如果节点不可达则为-1
func dijkstra(n, start int, edges []Edge) []float64 {
	ret := make([]float64, n) // 返回值

	// 各个节点都初始化为不可达
	for i := 0; i < n; i++ {
		ret[i] = unReachable
	}

	// 起点到起点的最短距离为0
	ret[start] = 0

	// 构建相邻边
	neighbours := makeNeighbours(n, edges)

	// 创建小顶堆
	var hp MinHeap
	heap.Init(&hp)
	heap.Push(&hp, Item{start, 1}) // 下标0表示节点id，下标1表示从起点到该节点的最短距离

	for hp.Len() > 0 {
		item := heap.Pop(&hp).(Item)
		var (
			id  = item.id     // 当前节点id
			dis = item.weight // 从起点到当前节点的最短距离
		)
		//fmt.Println("id: ", id, " dis: ", dis)

		// 如果之前有更小的路径到达该节点，则跳过
		if ret[id] != unReachable && dis < ret[id] {
			continue
		}

		// 走到这说明要么是第一次到达该节点，要么当前这条路径是达到该点的最短距离
		ret[id] = dis

		// 将该节点的相邻节点都加入到小顶堆中
		for _, v := range neighbours[id] {
			// 如果相邻节点之前未到达过 或者 当前这条路径比之前的任何路径的距离都要短
			w := ret[id] * v.weight
			if ret[v.to] == unReachable || w > ret[v.to] { // 加上这个条件实际上是过滤掉了那些永远不可能成为最短距离的路径，达到了剪枝的效果
				heap.Push(&hp, Item{v.to, w})
			}
		}
	}
	return ret
}

// makeNeighbours 构建相邻边
func makeNeighbours(n int, edges []Edge) [][]Edge {
	neighbours := make([][]Edge, n)
	for _, v := range edges {
		neighbours[v.from] = append(neighbours[v.from], v)
	}
	return neighbours
}

// MinHeap 大顶堆
type MinHeap []Item

func (pq MinHeap) Len() int { return len(pq) }

func (pq MinHeap) Less(i, j int) bool {
	return pq[i].weight > pq[j].weight
}

func (pq MinHeap) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}

func (pq *MinHeap) Push(x interface{}) {
	e := x.(Item)
	*pq = append(*pq, e)
}

func (pq *MinHeap) Pop() interface{} {
	old := *pq
	n := len(old)
	node := old[n-1]
	*pq = old[:n-1]
	return node
}
```