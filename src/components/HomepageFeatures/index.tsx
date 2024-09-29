import clsx from "clsx";
import Heading from "@theme/Heading";
import styles from "./styles.module.css";

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<"svg">>;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: "高效编程",
    Svg: require("@site/static/img/undraw_docusaurus_mountain.svg").default,
    description: (
      <>
        简化专用内核开发，编写 GPU
        代码更高效，而无需深入了解复杂的加速计算框架。
      </>
    ),
  },
  {
    title: "实时编译",
    Svg: require("@site/static/img/undraw_docusaurus_tree.svg").default,
    description: (
      <>支持即时编译，允许动态生成和优化 GPU 代码，适应不同硬件和任务需求。</>
    ),
  },
  {
    title: "灵活的迭代空间结构",
    Svg: require("@site/static/img/undraw_docusaurus_react.svg").default,
    description: (
      <>
        采用分块程序和标量线程，增强了迭代空间的灵活性，便于处理稀疏操作和优化数据局部性。
      </>
    ),
  },
];

function Feature({ title, Svg, description }: FeatureItem) {
  return (
    <div className={clsx("col col--4")}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
