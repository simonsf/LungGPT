import json


def save_json(save_path, data):
    """
    From saving dictionary at json format
    :param save_path:
    :param data:
    :return:
    """
    with open(save_path, 'w', encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2))


def find_value_in_nested_dict(data, key):
  """
  在嵌套字典中查找指定key对应的值

  Args:
    data: 嵌套字典
    key: 要查找的key

  Returns:
    找到则返回对应的值，否则返回None
  """

  if isinstance(data, dict):
    for k, v in data.items():
      if k == key:
        return v
      elif isinstance(v, (dict, list)):
        result = find_value_in_nested_dict(v, key)
        if result:
          return result
  return None


def common_treatment_key(class_dict, disease):
   """
   解决一些亚类没有，但有大类的公共治疗方法
   param class_dict: 和诊疗指南吻合的疾病类别字典
   param disease: 当前疾病亚类名
   """
   if isinstance(class_dict, dict):
    for k, v in class_dict.items():
      # if k == disease:
      #   print("Wrong search")
      #   return disease
      if isinstance(v, list):
        if disease in v:
          return k
      elif isinstance(v, dict):
        if disease in v.keys():
          return k
        result = common_treatment_key(v, disease)
        if result:
          return result
   
   return None


def get_treatment(treatment_dict, class_dict, disease):
    """
    解决一些亚类没有，但有大类的公共治疗方法
    param treatment_dict: 治病治疗信息字典
    param disease: 当前疾病亚类名
    """
    treatment = find_value_in_nested_dict(treatment_dict, disease)
    if treatment:
      return treatment
    else:
      disease = common_treatment_key(class_dict, disease)
      treatment = get_treatment(treatment_dict, class_dict, disease)
      if treatment:
        return treatment
    return None


def treatment_dict2str(treatment_dict):
  """
  Fromating 
  """  
  if isinstance(treatment_dict, dict):
      if list(treatment_dict.keys())[0] == "治疗":
        treatment_dict = treatment_dict["治疗"]
      # return "{" + ",".join(f"{key}: {treatment_dict2str(value)}" for key, value in treatment_dict.items()) + "}"
      return  ",".join(f"{key}: {treatment_dict2str(value)}" for key, value in treatment_dict.items())
  elif isinstance(treatment_dict, list):
      # return "[" + ",".join(treatment_dict2str(item) for item in treatment_dict) + "]"
      return ",".join(treatment_dict2str(item) for item in treatment_dict)
  else:
      return str(treatment_dict)
