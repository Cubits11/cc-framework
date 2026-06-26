#!/usr/bin/env node
import "source-map-support/register";
import * as cdk from "aws-cdk-lib";
import { CcEnterpriseStack } from "../lib/cc-enterprise-stack";

const app = new cdk.App();

new CcEnterpriseStack(app, "CcEnterpriseReferenceStack", {
  description: "Minimum credible CC Framework AWS reference architecture",
});
