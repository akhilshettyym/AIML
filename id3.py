import pandas as pd
df_tennis = pd.read_csv('/home/student/4.csv')
print('Given play tennis data set :\n\n',df_tennis)

def entropy(probs):
    import math
    return sum([-prob*math.log(prob,2)for prob in probs])

def entropy_of_list(a_list):
    from collections import Counter
    cnt = Counter(x for x in a_list)
    num_instances = len(a_list)*1.0
    probs = [x/num_instances for x in cnt.values()]
    return entropy(probs)

def information_gain(df,split_attribute_name,target_attribute_name,trace=0):
    df_split = df.groupby(split_attribute_name)
    nobs = len(df.index)*1.0
    df_agg_ent = df_split.agg ({target_attribute_name : [entropy_of_list,lambda x :len(x)/nobs] })[target_attribute_name]
    df_agg_ent.columns = ['Entropy','PropObservations']
    new_entropy = sum(df_agg_ent['Entropy']*df_agg_ent['PropObservations'])
    old_entropy = entropy_of_list(df[target_attribute_name])
    return old_entropy - new_entropy

def id3(df,target_attribute_name,attribute_names,default_class=None):
    from collections import Counter
    cnt = Counter(x for x in df[target_attribute_name])
    print(cnt)
    if len(cnt)==1:
        print(len(cnt))
        return next(iter(cnt))
    elif df.empty or (not attribute_names):
        return default_class
    else:
        default_class = max(cnt.keys())
        gainz = [information_gain(df,attr, target_attribute_name) for attr in attribute_names]
        print('Gain :',gainz)
        index_of_max = gainz.index(max(gainz))
        best_attr = attribute_names[index_of_max]
        print('Best attribute :',best_attr)
        tree = {best_attr :{}}
        remaining_attribute_name = [i for i in attribute_names if i!=best_attr]
        for attr_val, data_subset in df.groupby(best_attr):
            subtree = id3(data_subset,
                          target_attribute_name,
                          remaining_attribute_name,
                          default_class)
            tree[best_attr][attr_val] = subtree
        return tree
    
attribute_names = list(df_tennis.columns)
print('List of attributes :',attribute_names)
attribute_names.remove('Play Tennis')
print('Predicting attributes',attribute_names)

from pprint import pprint
tree = id3(df_tennis,'Play Tennis', attribute_names)
print('The resultant attribute is :\n')
pprint(tree)

def classify(instance,tree,default=None):
    attribute = next(iter(tree))

    if instance[attribute] in tree[attribute].keys():
        result = tree[attribute][instance[attribute]]
        if isinstance(result,dict):
            return classify(instance,result)
        else:
            return result
    else:
        return default
    
test_data = pd.read_csv('/home/student/4.csv')
test_data['predicted2'] = test_data.apply(classify,axis=1,args=(tree,''))

print(test_data[['predicted2']])