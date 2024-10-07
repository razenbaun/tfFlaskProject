from requests import get
sepal_length = input('Введите рост = ')
sepal_width = input('Введите вес = ')
petal_length = input('Введите размер обуви = ')
petal_width = input('Введите возраст = ')
print(get('http://localhost:5000/api_v2', json={'list1': sepal_length,
                                                'list2': sepal_width,
                                                'list3': petal_length,
                                                'list4': petal_width}).json())