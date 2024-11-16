import tree
import settings as stgs
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # Загрузка тренировочной и тестовой выборок
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                        train_size=stgs.TRAIN_SIZE, random_state=stgs.RANDOM_STATE)

    # Тренировка дерева
    tree_model = tree.Tree()
    tree_model.train(x_train, y_train)

    # Вывод результатов
    accurancy = 0

    print(f"|{"Правильные решения":^20}|{"Решения дерева":^20}|")
    for i, j in zip(y_test, x_test):
        actual = str(iris.target_names[i])
        prediction = str(iris.target_names[tree_model.root_node.get_decision(j)])
        print(f"|{actual:^20}|{prediction:^20}|")

        if actual == prediction:
            accurancy += 1

    print(f"\nТочность предсказаний модели: {(accurancy/y_test.size)*100:02.2f}%")





