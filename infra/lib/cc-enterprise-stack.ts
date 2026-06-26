import * as path from "path";
import {
  Duration,
  RemovalPolicy,
  Stack,
  StackProps,
  CfnOutput,
} from "aws-cdk-lib";
import * as apigateway from "aws-cdk-lib/aws-apigateway";
import * as dynamodb from "aws-cdk-lib/aws-dynamodb";
import * as iam from "aws-cdk-lib/aws-iam";
import * as kms from "aws-cdk-lib/aws-kms";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as s3 from "aws-cdk-lib/aws-s3";
import { Construct } from "constructs";

export class CcEnterpriseStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    const evidenceBucket = new s3.Bucket(this, "EvidenceBundleBucket", {
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
      objectLockEnabled: true,
      removalPolicy: RemovalPolicy.RETAIN,
      versioned: true,
    });
    const bucketResource = evidenceBucket.node.defaultChild as s3.CfnBucket;
    bucketResource.objectLockConfiguration = {
      objectLockEnabled: "Enabled",
      rule: {
        defaultRetention: {
          mode: "COMPLIANCE",
          days: 30,
        },
      },
    };

    const attestationKey = new kms.CfnKey(this, "AttestationSigningKey", {
      description: "Asymmetric KMS key for CC evidence attestation signing and verification",
      enabled: true,
      keySpec: "RSA_2048",
      keyUsage: "SIGN_VERIFY",
      keyPolicy: {
        Version: "2012-10-17",
        Statement: [
          {
            Sid: "EnableAccountAdministration",
            Effect: "Allow",
            Principal: { AWS: `arn:aws:iam::${this.account}:root` },
            Action: "kms:*",
            Resource: "*",
          },
        ],
      },
    });

    const metadataTable = new dynamodb.Table(this, "RunMetadataTable", {
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      partitionKey: { name: "bundle_id", type: dynamodb.AttributeType.STRING },
      removalPolicy: RemovalPolicy.RETAIN,
      sortKey: { name: "record_type", type: dynamodb.AttributeType.STRING },
    });

    const verifierRole = new iam.Role(this, "VerifierLambdaRole", {
      assumedBy: new iam.ServicePrincipal("lambda.amazonaws.com"),
      description: "Least-privilege role for verify(bundle_id).",
    });
    verifierRole.addManagedPolicy(
      iam.ManagedPolicy.fromAwsManagedPolicyName("service-role/AWSLambdaBasicExecutionRole"),
    );
    verifierRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ["s3:GetObject", "s3:GetObjectVersion"],
        resources: [evidenceBucket.arnForObjects("bundles/*")],
      }),
    );
    verifierRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ["dynamodb:GetItem"],
        resources: [metadataTable.tableArn],
      }),
    );
    verifierRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ["kms:Verify", "kms:GetPublicKey"],
        resources: [attestationKey.attrArn],
      }),
    );

    const evidenceWriterRole = new iam.Role(this, "EvidenceWriterRole", {
      assumedBy: new iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
      description:
        "Least-privilege role for an evaluation task that writes signed evidence bundles.",
    });
    evidenceWriterRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ["s3:PutObject", "s3:PutObjectRetention", "s3:AbortMultipartUpload"],
        resources: [evidenceBucket.arnForObjects("bundles/*")],
      }),
    );
    evidenceWriterRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ["dynamodb:PutItem", "dynamodb:UpdateItem"],
        resources: [metadataTable.tableArn],
      }),
    );
    evidenceWriterRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ["kms:Sign", "kms:GetPublicKey"],
        resources: [attestationKey.attrArn],
      }),
    );

    const verifier = new lambda.Function(this, "VerifyBundleFunction", {
      code: lambda.Code.fromAsset(path.join(__dirname, "..", "lambda")),
      description: "Single backend endpoint for verify(bundle_id).",
      environment: {
        ATTESTATION_KEY_ID: attestationKey.ref,
        EVIDENCE_BUCKET: evidenceBucket.bucketName,
        RUN_METADATA_TABLE: metadataTable.tableName,
      },
      handler: "verify_handler.handler",
      memorySize: 256,
      role: verifierRole,
      runtime: lambda.Runtime.PYTHON_3_12,
      timeout: Duration.seconds(10),
    });

    const api = new apigateway.RestApi(this, "VerifyApi", {
      description: "Minimal CC evidence verification API.",
      deployOptions: {
        metricsEnabled: true,
        tracingEnabled: true,
      },
    });
    const verify = api.root.addResource("verify");
    const bundle = verify.addResource("{bundle_id}");
    bundle.addMethod("GET", new apigateway.LambdaIntegration(verifier));

    new CfnOutput(this, "EvidenceBucketName", {
      value: evidenceBucket.bucketName,
    });
    new CfnOutput(this, "RunMetadataTableName", {
      value: metadataTable.tableName,
    });
    new CfnOutput(this, "AttestationKeyId", {
      value: attestationKey.ref,
    });
    new CfnOutput(this, "VerifyEndpoint", {
      value: `${api.url}verify/{bundle_id}`,
    });
    new CfnOutput(this, "EvidenceWriterRoleArn", {
      value: evidenceWriterRole.roleArn,
    });
  }
}
