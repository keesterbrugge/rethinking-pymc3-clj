;; https://github.com/pymc-devs/resources/blob/master/Rethinking_2/Chp_05.ipynb
;; https://fehiepsi.github.io/rethinking-numpyro/05-the-many-variables-and-the-spurious-waffles.html

(ns rethinking-pymc3-clj.ch5
 (:require [libpython-clj.python :as py :refer [py.]]
           [libpython-clj.require :refer [require-python]]
           [applied-science.darkstar :as darkstar]
           [tech.ml.dataset :as ds]
           [tech.ml.dataset.pipeline :as dsp]  
           [tech.v2.datatype :as datatype]
           [tech.v2.tensor :as tensor]
           [clojure.java.shell :as sh]))



(require-python 'builtins
                '[pymc3 :as pm :bind-ns]
                '[numpy :as np :bind-ns]
                '[pymc3.math :as pm-math :bind-ns]
                '[arviz :as az]
                '[pandas :as pd]
                '[scipy.stats :as stats :refer [bernoulli beta]]
                'operator)



(defn data []
  (ds/->dataset "https://raw.githack.com/rmcelreath/rethinking/master/data/WaffleDivorce.csv"
                {:separator \;}))

(defn processed-ds []
  (-> (data)
      (dsp/std-scale  ["Marriage" "MedianAgeMarriage" "Divorce"])))

(comment
  (ds/descriptive-stats (data))
  (processed-ds)
  )




(defmacro quick-trace
  [& body]
  ;; TODO can I add something that needs to be passed back besides trace, like prior predictive? 
  (let [
        ;; bindings (partition 2 body)
        ;; bindings' (mapcat (fn [[symb ls]]
                            ;; [symb (concat (list(first ls) (name symb)) (rest ls))]) bindings)
        bindings (->> (partition 2 body)
                        (mapcat (fn [[symb [dist & dist-args]]]
                                  [symb (concat (list dist (name symb))
                                                dist-args)])))]
    #_(println bindings')
    (println bindings)
    `(py/with
      [_# (pm/Model)]
      (let [~@bindings]
        (pm/sample 2000)))))

(comment 
  (macroexpand-1 '(quick-trace
                   a (pm/Normal :mu 0 :sigma 0.2) 
                   bA (pm/Normal :mu 0 :sigma 0.5)
                   sigma (pm/Exponential :lam 1)
                   mu (pm/Deterministic (operator/add a
                                                      (operator/mul bA ((processed-ds) "MedianAgeMarriage"))))
                   divorce_rate_std (pm/Normal :mu mu :sigma sigma :observed ((processed-ds) "Divorce"))
                   )))


(comment 
  (def trace (quick-trace
              a (pm/Normal :mu 0 :sigma 0.2) 
              bA (pm/Normal :mu 0 :sigma 0.5)
              sigma (pm/Exponential :lam 1)
              mu (pm/Deterministic (operator/add  a
                                                  (operator/mul bA ((processed-ds) "MedianAgeMarriage"))))
              divorce_rate_std (pm/Normal :mu mu :sigma sigma :observed ((processed-ds) "Divorce")))))


;; az.plot_trace(m_5_1_trace, var_names=["a", "bA"]);


;; (require '[clojure.java.shell :as sh])
;; (def mplt (py/import-module "matplotlib"))
;; (py. mplt "use" "Agg")
;; (require-python 'matplotlib.pyplot)
;; (require-python 'matplotlib.backends.backend_agg)

(defmacro with-show
  "Takes forms with mathplotlib.pyplot to then show locally"
  [& body]
  (require '[clojure.java.shell :as sh])
  (def mplt (py/import-module "matplotlib"))
  (py. mplt "use" "Agg")
  (require-python 'matplotlib.pyplot)
  (require-python 'matplotlib.backends.backend_agg)

  `(let [_# (matplotlib.pyplot/clf)
         fig# (matplotlib.pyplot/figure)
         agg-canvas# (matplotlib.backends.backend_agg/FigureCanvasAgg fig#)]
     ~(cons 'do body)
     (py. agg-canvas# "draw")
     (matplotlib.pyplot/savefig "temp.png")
     (sh/sh "open" "temp.png")))

(comment 
  (with-show (az/plot_trace trace :var_names ["a" "bA"])))


(defn trace->dataset [trace]
  (->> trace
       np/array
       (map py/->jvm)
       (map #(reduce-kv (fn [m k v] (assoc m k (if (tech.v2.tensor/tensor? v) (vec v) v))) {} %))
       ds/->dataset))

(defmacro quick-trace3
  [& body]
  ;; TODO can I add something that needs to be passed back besides trace, like prior predictive? 
  (let [bindings (->> (partition 2 body)
                      (mapcat (fn [[symb [dist & dist-args]]]
                                [symb `(~dist ~(name symb) ~@dist-args)]
                                #_[symb (concat (list dist (name symb))
                                              dist-args)])))]
    (println bindings)
    `(py/with
      [_# (pm/Model)]
      (let [~@bindings
            trace# (pm/sample 2000)]
        {:trace trace#
         :sample-dataset (trace->dataset trace#)
         :prior-pred-sample (pm/sample_prior_predictive)
         :posterior-pred-sample (pm/sample_posterior_predictive trace# :samples 1000)}))))


(comment 
  (macroexpand-1 '(quick-trace3
                   a (pm/Normal :mu 0 :sigma 0.2) 
                   bA (pm/Normal :mu 0 :sigma 0.5)
                   sigma (pm/Exponential :lam 1)
                   mu (pm/Deterministic (operator/add  a
                                                       (operator/mul bA ((processed-ds) "MedianAgeMarriage"))))
                   divorce_rate_std (pm/Normal :mu mu :sigma sigma :observed ((processed-ds) "Divorce")))))

(def trace3 (quick-trace3
             a (pm/Normal :mu 0 :sigma 0.2) 
             bA (pm/Normal :mu 0 :sigma 0.5)
             sigma (pm/Exponential :lam 1)
             mu (pm/Deterministic (operator/add  a
                                                 (operator/mul bA ((processed-ds) "MedianAgeMarriage"))))
             divorce_rate_std (pm/Normal :mu mu :sigma sigma :observed ((processed-ds) "Divorce"))))


(comment 
  (-> trace3
      :trace
      (az/plot_trace :var_names ["a" "bA"])
      with-show))



(defn view! [spec]
  (->> (clojure.data.json/json-str spec)
       darkstar/vega-lite-spec->svg
       (spit "tmp-darkstar.svg"))
  (sh/sh "qlmanage" "-p" "tmp-darkstar.svg")
  ;; (clojure.java.browse/browse-url "file:://tmp/darkstar.svg")
  )


(let [param-maps (-> trace3
          :sample-dataset
          (ds/select ["a" "bA"] (range 30))
          ds/->flyweight)
      data-map (vec (for [{:strs [a bA] :as parameters} param-maps
                          x (range -2 3)]
                      {:detail (str parameters)
                       :x x
                       :y (+ a (* bA x))}))
      spec {:data {:values data-map }
            :mark :line
            :encoding {:x {:field :x}
                       :y {:field :y}
                       :detail {:field :detail :type :nominal}}}]
  (view! spec))

(defn trace->dataset [trace]
  (->> trace
       np/array
       (map py/->jvm)
       (map #(reduce-kv (fn [m k v] (assoc m k (if (tech.v2.tensor/tensor? v) (vec v) v))) {} %))
       ds/->dataset))


(let [ds (->> trace3
              :prior-pred-sample
              py/->jvm
              (reduce-kv (fn [m k v] (assoc m k (if (tech.v2.tensor/tensor? v) (vec v) v))) {} )
              ds/name-values-seq->dataset)
      ]
  ;; (ds/->flyweight (ds/select ds ["divorce_rate_std"] 10))
  (take 2 (datatype/->reader (ds "divorce_rate_std")))
  (mapv #(hash-map :detail %1 :x %2 :y %3) (range) ((processed-ds) "MedianAgeMarriage") (first (datatype/->reader (ds "divorce_rate_std"))) )
  )



(-> trace3
     :prior-pred-sample
     py/->jvm
     (dissoc "divorce_rate_std")
     ds/name-values-seq->dataset)






(->> trace3
    :prior-pred-sample
    py/->jvm
    #_(take 2)
    #_merge
    #_(into {})
    (reduce-kv (fn [m k v] (assoc m k (if (tech.v2.tensor/tensor? v) (vec v) v))) {} )
    ds/name-values-seq->dataset
    #_(get "divorce_rate_std")
    )


(ds/->flyweight (ds/name-values-seq->dataset {:a [0 1] :ary [[1 2 3] [4 5 6]]}))
;; => ({:a 0.0, :ary [1 2 3]} {:a 1.0, :ary [4 5 6]})

(ds/name-values-seq->dataset {:a [0 1] :ary (tensor/->tensor [[1 2 3] [4 5 6]])})


(ds/name-values-seq->dataset {:a [0 1] :ary (tensor/->tensor [[2 3] [4 5]])})
    

(ds/->dataset [{:a 0 :tensor (tech.v2.tensor/->tensor [1 2 3])}
               {:a 1 :tensor (tech.v2.tensor/->tensor [4 5 6])}])

(ds/name-values-seq->dataset {:a [0 1]
                              :tensor (tech.v2.tensor/->tensor [[1 2 3]
                                                                [4 5 6]])})

(= (ds/->dataset [{:a 0 :tensor (tech.v2.tensor/->tensor [1 2 3])}
                  {:a 1 :tensor (tech.v2.tensor/->tensor [4 5 6])}])

   (ds/name-values-seq->dataset {:a [0 1]
                                 :tensor (tech.v2.tensor/->tensor [[1 2 3]
                                                                   [4 5 6]])}))
